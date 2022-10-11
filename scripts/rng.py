import torch
import numpy as np


class BatchCudaRNG:
    is_batch = True

    def __init__(self, device, shape, tch_dtype):
        self.device = device

        self.shape = shape
        self.shape_prod = int(np.prod(self.shape))
        self.shape_len = len(self.shape)
        self.reset_shape_attrs(shape)

        self.rngs = [torch.Generator(device=self.device) for _ in range(self.shape_prod)]
        self.tch_dtype = tch_dtype

        self.unif_cache_cols = 1_000_000
        self.unif_cache = torch.empty((self.shape_prod, self.unif_cache_cols),
                                      device=self.device, dtype=tch_dtype)
        self.unif_cache_col_idx = self.unif_cache_cols  # So that it would get refilled immediately
        self.unif_cache_rng_states = None

        self.norm_cache_cols = 5_000_000
        self.norm_cache = torch.empty((self.shape_prod, self.norm_cache_cols),
                                      device=self.device, dtype=tch_dtype)
        self.norm_cache_col_idx = self.norm_cache_cols  # So that it would get refilled immediately
        self.norm_cache_rng_states = None

    def reset_shape_attrs(self, shape):
        self.shape = shape
        self.shape_prod = int(np.prod(self.shape))
        self.shape_len = len(self.shape)

    def seed(self, seed_arr):
        # Collecting the rng_states after seeding
        assert isinstance(seed_arr, np.ndarray)
        assert len(self.rngs) == seed_arr.size
        flat_seed_arr = seed_arr.copy().reshape(-1)
        np_random = np.random.RandomState(seed=0)
        for seed, rng in zip(flat_seed_arr, self.rngs):
            np_random.seed(seed)
            balanced_32bit_seed = np_random.randint(0, 2**31-1, dtype=np.int32)
            rng.manual_seed(int(balanced_32bit_seed))

        if self.unif_cache_col_idx < self.unif_cache_cols:
            self.refill_unif_cache()
            # The cache has been used before, so in order to be able to
            # concat this sampler with the non-reseeded sampler, we should not
            # change the self.unif_cache_cols.

            # Note: We should not refill the uniform cache if the model
            # has not been initialized. This is done to keep the backward
            # compatibility and reproducibility properties with the old scripts.
            # Otherwise, the order of random samplings will change. Remember that
            # the old script first uses dirichlet and priors, and then refills
            # the unif/norm cache. In order to be similar, we should avoid
            # refilling the cache upon the first .seed() call
        if self.norm_cache_col_idx < self.norm_cache_cols:
            self.refill_norm_cache()

    def get_state(self):
        state_dict = dict(unif_cache_rng_states=self.unif_cache_rng_states,
                          norm_cache_rng_states=self.norm_cache_rng_states,
                          norm_cache_col_idx=self.norm_cache_col_idx,
                          unif_cache_col_idx=self.unif_cache_col_idx,
                          rng_states=self.get_rng_states(self.rngs))
        return state_dict

    def set_state(self, state_dict):
        unif_cache_rng_states = state_dict['unif_cache_rng_states']
        norm_cache_rng_states = state_dict['norm_cache_rng_states']
        norm_cache_col_idx = state_dict['norm_cache_col_idx']
        unif_cache_col_idx = state_dict['unif_cache_col_idx']
        rng_states = state_dict['rng_states']

        if unif_cache_rng_states is not None:
            self.set_rng_states(unif_cache_rng_states, self.rngs)
            self.refill_unif_cache()
            self.unif_cache_col_idx = unif_cache_col_idx
        else:
            self.unif_cache_col_idx = self.unif_cache_cols
            self.unif_cache_rng_states = None

        if norm_cache_rng_states is not None:
            self.set_rng_states(norm_cache_rng_states, self.rngs)
            self.refill_norm_cache()
            self.norm_cache_col_idx = norm_cache_col_idx
        else:
            self.norm_cache_col_idx = self.norm_cache_cols
            self.norm_cache_rng_states = None

        self.set_rng_states(rng_states, self.rngs)

    def get_rngs(self):
        return self.rngs

    def set_rngs(self, rngs, shape):
        assert isinstance(rngs, list)
        self.reset_shape_attrs(shape)
        self.rngs = rngs
        assert len(self.rngs) == self.shape_prod, f'{len(self.rngs)} != {self.shape_prod}'

    def get_rng_states(self, rngs):
        """
        getting state in ByteTensor
        """
        rng_states = []
        for i, rng in enumerate(rngs):
            rng_state = rng.get_state()
            rng_states.append(rng_state.detach().clone())
        return rng_states

    def set_rng_states(self, rng_states, rngs):
        """
        rng_states should be ByteTensor (RNG state must be a torch.ByteTensor)
        """
        assert isinstance(rng_states, list), f'{type(rng_states)}, {rng_states}'
        for i, rng in enumerate(rngs):
            rng.set_state(rng_states[i].cpu())

    def __call__(self, gen, sample_shape):
        sample_shape_rightmost = sample_shape[self.shape_len:]
        random_vars = []
        for i, rng in enumerate(self.rngs):
            rng_state = rng.get_state().detach().clone()
            torch.cuda.set_rng_state(rng_state, self.device)
            random_vars.append(gen.sample(sample_shape_rightmost))
            rng.set_state(torch.cuda.get_rng_state(self.device).detach().clone())
        rv = torch.stack(random_vars, dim=0).reshape(*sample_shape)
        return rv

    def dirichlet(self, gen_list, sample_shape):
        sample_shape_rightmost = sample_shape[self.shape_len:]
        random_vars = []
        for i, (gen_, rng) in enumerate(zip(gen_list, self.rngs)):
            rng_state = rng.get_state().detach().clone()
            torch.cuda.set_rng_state(rng_state, self.device)
            random_vars.append(gen_.sample(sample_shape_rightmost))
            rng.set_state(torch.cuda.get_rng_state(self.device).detach().clone())

        rv = torch.stack(random_vars, dim=0)
        rv = rv.reshape(*self.shape, *rv.shape[1:])
        return rv

    def refill_unif_cache(self):
        self.unif_cache_rng_states = self.get_rng_states(self.rngs)
        for row, rng in enumerate(self.rngs):
            self.unif_cache[row].uniform_(generator=rng)

    def refill_norm_cache(self):
        self.norm_cache_rng_states = self.get_rng_states(self.rngs)
        for row, rng in enumerate(self.rngs):
            self.norm_cache[row].normal_(generator=rng)

    def uniform(self, sample_shape):
        sample_shape_tuple = tuple(sample_shape)
        assert sample_shape_tuple[:self.shape_len] == self.shape

        sample_shape_rightmost = sample_shape[self.shape_len:]
        cols = np.prod(sample_shape_rightmost)
        if self.unif_cache_col_idx + cols >= self.unif_cache_cols:
            self.refill_unif_cache()
            self.unif_cache_col_idx = 0

        samples = self.unif_cache[:, self.unif_cache_col_idx: (self.unif_cache_col_idx + cols)]
        samples = samples.reshape(*sample_shape)
        self.unif_cache_col_idx += cols

        return samples

    def normal(self, sample_shape):
        sample_shape_tuple = tuple(sample_shape)
        cols = np.prod(sample_shape_tuple) // self.shape_prod
        assert cols * self.shape_prod == np.prod(sample_shape_tuple)
        if self.norm_cache_col_idx + cols >= self.norm_cache_cols:
            self.refill_norm_cache()
            self.norm_cache_col_idx = 0

        samples = self.norm_cache[:, self.norm_cache_col_idx: (self.norm_cache_col_idx + cols)]
        samples = samples.reshape(*sample_shape)
        self.norm_cache_col_idx += cols

        return samples

    @classmethod
    def Merge(cls, sampler1, sampler2):
        assert sampler1.shape_len == sampler2.shape_len == 1

        device = sampler1.device
        tch_dtype = sampler1.tch_dtype
        chain_size = (sampler1.shape[0]+sampler2.shape[0],)

        state_dict1, state_dict2 = sampler1.get_state(), sampler2.get_state()

        merged_state_dict = dict()
        for key in state_dict1:
            if key in ('unif_cache_rng_states', 'norm_cache_rng_states',
                       'rng_states'):
                # saba modified
                if (state_dict1[key] is None) and (state_dict2[key] is None):
                    merged_state_dict[key] = None
                elif (state_dict1[key] is None) or (state_dict2[key] is None):
                    raise ValueError(f"{key} with None occurance")
                else:
                    merged_state_dict[key] = state_dict1[key] + state_dict2[key]
            elif key in ('norm_cache_col_idx', 'unif_cache_col_idx'):
                assert state_dict1[key] == state_dict2[key]
                merged_state_dict[key] = state_dict1[key]
            else:
                raise ValueError(f'Unknown rule for {key}')

        sampler = cls(device, chain_size, tch_dtype)
        sampler.set_state(merged_state_dict)
        return sampler

    @classmethod
    def Subset(cls, sampler, chain_inds):
        assert sampler.shape_len == 1

        device = sampler.device
        tch_dtype = sampler.tch_dtype
        chain_size_sub = (len(chain_inds),)

        state_dict = sampler.get_state()

        sub_state_dict = dict()
        for key in state_dict:
            if key in ('unif_cache_rng_states', 'norm_cache_rng_states',
                       'rng_states'):
                sub_state_dict[key] = [state_dict[key][ind] for ind in chain_inds]
            elif key in ('norm_cache_col_idx', 'unif_cache_col_idx'):
                sub_state_dict[key] = state_dict[key]
            else:
                raise ValueError(f'Unknown rule for {key}')

        sampler = cls(device, chain_size_sub, tch_dtype)
        sampler.set_state(sub_state_dict)
        return sampler


class SerialRNG:
    is_batch = False

    def __init__(self):
        pass

    def __call__(self, gen, sample_shape):
        return gen.sample(sample_shape=sample_shape)
