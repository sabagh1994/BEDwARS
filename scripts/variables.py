class RV:
    def __init__(self, name, value=None, distribution=None,
                 dependency=None, type_=None, size=None, is_transform=False):
        self.type_ = type_  # 'free' or 'fixed'
        self.name = name
        self.value = value
        self.size = size
        self.dependency = dependency
        self.distribution = distribution
        self.is_transform = is_transform

    def logp(self, value, point=None):
        return self.distribution.logp(value, point)

    def init_value(self, point=None, seed=None, init_value=None, is_transformed=False, do_prior_init=True):
        """
        Initialize the variable value
        :param point: (dict) contains the distribution parameters used for the variable
        :param seed: not used (set to None)
        :param init_value: (torch.Tensor) The initialization value used for the variable
        :param is_transformed: (boolean) Specifies whether the provided initialization value (init_value)
                                was already transformed or not (forward transformation).
                                This is because the space of some variables should be transformed prior
                                to sampling with Metropolis Hastings. If the user provides a non-transformed
                                value then this should be set to False.
        :param do_prior_init: (boolean) Whether to use variable prior for initialization
        :return:
        """
        if init_value is not None:
            print(f'manual initialization on variable {self.name}\n')
            self.value = init_value
            if self.is_transform and (not is_transformed):
                print('forward transformation in initialization')
                self.value = self.distribution.transform.forward(init_value)
        elif do_prior_init:
            self.value = self.distribution.generate_samples(size=self.size, point=point, seed=seed)
            print(f"prior initialization for variable {self.name}\n")
        else:
            print(f"no initialization performed on variable {self.name}\n")

    def backward_var(self, value=None):
        """
        Perform the backward transformation on the value
        :param value: (torch.Tensor) The value to be transformed depending on the variable transformation
        :return: (torch.Tensor) Transformed or non-transformed value
        """
        if (not self.is_transform) and (value is None):
            return self.value
        elif (not self.is_transform) and (not(value is None)):
            return value
        elif value is None:
            return self.distribution.transform.backward(self.value)
        return self.distribution.transform.backward(value)

    def size_(self):
        if not self.size:  # torch.tensor(1.).shape => []
            return 1
        return self.value.reshape(-1).shape[0]
