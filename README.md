# Robust <ins>B</ins>ayesian <ins>E</ins>xpression <ins>D</ins>econvolution <ins>w</ins>ith <ins>A</ins>pproximate <ins>R</ins>eference <ins>S</ins>ignatures (BEDwARS)

BEDwARS is a Bayesian approach to deconvolving bulk expression profiles using reference expression profiles (“signatures”) of the constituent cell types. It is designed to be robust to “noise” in provided reference signatures that may arise due to biological and/or technical differences from the bulk expression profiles. 


<details open>
<summary><h2>Guide to BEDwARS</h2></summary>

+  <details>
   <summary><strong>Cloning the Repository</strong></summary>
   
    1. `git clone --recursive https://github.com/sabagh1994/BEDwARS.git`
    2. `cd BEDwARS`
    </details>

+  <details>
   <summary><strong>Download the Datasets and Configs</strong></summary>
   
   To download the signatures, mixtures, and proportions used in the benchamrking and deconvolution of DPD deficiency, 
   run `./input/download.sh`. The files will be located at `./input/signatures`, `./input/mixtures`, and `./input/proportions`.
   
   To download the config files run `./configs/download.sh`. The configs will be stored at `configs/brain` (benchmark on brain, 
   `./configs/pancreas` (benchmark on pancreas), and `./configs/dpd_brain` (DPD deficiency).
    </details>

+  <details>
   <summary><strong>Make a Virtual Environment</strong></summary>
   
   Before running BEDwARS make sure that all the required packages are installed.
   To create a virtual environment with all the required packages installed,
   
    1. Install Python if you do not have it. We used Python 3.8.
    2. Run `make venv`. This step creates a folder named `./venv` which contains all the required packages.
    3. Run `source venv/bin/activate` to activate the venv
    </details>

+  <details>
   <summary><strong>Running BEDwARS</strong></summary>
   
   To run BEDwARS two input files in json format are needed. One of the files `./configs/cfg_sample.json` contains the path to reference signature and 
   the bulk expression profiles. Read more in **Configurations** below. The other file `./configs/curric.json` contains the set of instructions, called
   curriculum, to initialize the model and sampler chains as well as running them. Read more in **Curriculum**.
   
   1. To run the metropolis hasting sampler, execute 
      ```bash
      python scripts/curriculum.py --general_config configs/cfg_sample.json \
             --curric_config configs/curric.json &> log_mh
      ```
      With the provided `./configs/cfg_sample.json` this script performs the deconvolution using Baron reference signature for pseudo-bulk 
      profiles generated from scRNA-seq data of Segerstolpe type II diabetic (T2D) samples. While running, the logs of the statistics 
      computed for the variables are written to `./results/seger_d/Baron/curric/status` and the checkpoints, containing the state of model and 
      metropolis hasting sampler, are saved at `./results/seger_d/Baron/curric/Model`. 
      This step takes time to complete. Read **Run time and Memory Requirement** for more details.
      After this step is finished, we need to estimate the parameters, i.e., signatures and proportions, using the best chain. 
   2. To infer the parameters, run 
      ```bash
      python scripts/inference.py --general_config configs/cfg_sample.json \
             --curric_config configs/curric.json --stage_name L3 --stage_range 59 160 &> log_infer
      ```
      The inference is done by averaging the variable values, i.e., model state, over a range of stages that the model was saved.
      The name of the stage is passed by `stage_name` (L3) and `stage_range` argument is used to pass the start (e.g., 59) and end (e.g., 160)
      stage indices. With the given arguments `"inference,py"` loads in the checkpoints `L3-59.pt`, ..., `L3-159.pt` and takes
      the average of the variable values over these checkpoints. Each checkpoint contains the model state which includes the variable values.
      For the given example (`configs/cfg_sample.json`), the inferred signatures and proportions will be stored at 
      `./results/seger_d/Baron/curric/aggr` in tab-delimited files. Read more about model saving and stage names in **Curriculum/Description of 
      the Stage Arguments**.
 
    </details>
</details>


<details>
<summary><h2>Configurations</h2></summary>

+ <details open>
  <summary><strong>Example</strong></summary>
  
   All the configurations used for benchmarking BEDwARS against other methods are in the `configs` folder separated by organ. 
   An example of the configuration file to `configs/cfg_sample.json` is,
   ```json
   {
    "description": "settings for preprocessing (e.g. transformation), input and output directories.",
    "transformation": "log",
    "marker_type": "all",
    "naive_marker_FC": 2.0,
    "marker_dir": null,
    "normalization": "mean",
    "ref_sig_dir": "./input/signatures/pancreas/Baron/Baron",
    "org_sig_dir": null,
    "mix_dir": "./input/mixtures/pancreas/emtab_d",
    "org_prop_dir": null,
    "outdir": "./results/emtabd/Baron"
   }
   ```
   </details>
 
+ <details>
  <summary><strong>Description of the Arguments</strong></summary>
   
   * `"description"` is the notes about the configuration file or whatever notes you want to keep for the configuration you are using.
   
   * `"transformation"` is the type of transformation applied to the reference signatures and bulk expression profiles as a preprocessing step
       before running BEDwARS. **This argument should be set to `"log"`**. The current version of the code does not support other transformations. 
       All the benchmarking experiments used "log" transformation.
   
   * `"marker_type"` is the criterion to choose the markers. It can be set to `"all"`, `"naive"` or `"provided"`. **Set this arguemnt to `"all"`
      which uses all the common genes between reference signature and bulk expression profiles**. 
   
      The other two values (`"naive"`, `"provided"`) can be used in future versions of the code where marker selection may become a part of 
      preprocessing pipeline. Briefly, `"naive"` mode takes the genes with at least X fold higher expression in one cell type than the others.        
      `"provided"` takes the marker gene set from input.
      
   *  `"naive_marker_FC"` is the value for X folds, which is explained above, in case `"naive"` is used as the `"marker_type"` argument. **Leave 
      this argument as it is, i.e. `"2.0"`, as the current version of the code does not support naive marker selection yet**.
   
   *  `"marker_dir"` is the directory to the file with markers saved in it. **Set this argument to `null` for the current version of the code**.
      This argument should be used when the markers are provided by the user, i.e., `"marker_type"` is set to `"provided"`.
   
   * `"narmalization"` is the type of normalization applied to each cell type signature profile and bulk expression profile after transformation.
      This argument can be set to `"mean"` and `"standard"`. **`"mean"` was used for all the experiments in the paper so it is best to leave it at
      that**.
   
   * `"ref_sig_dir"` path to a tab-separated file containing reference cell type signatures. The rows are genes and the columns are cell types. 
      The first column contains the gene names or ensemble ids and the rest of the columns are the cell types. Please tag all the columns including 
      the first column. See `./input/signatures/pancreas/Baron/Baron` for an example of reference signatue file.
   
   * `"org_sig_dir"` path to a tab-separated file containing the true cell type signatures if existed. **This argument should be set to `"null"` as 
      true cell type signatures are usually not available**. You can use it for method development purposes when true signatures or any other signatures    
      that you want to compare inferred signature to are available. 
      The file format is identical to the reference cell type signature. Read **Curriculum** to see how this argument is used
      while running BEDwRAS. 
   
   * "`mix_dir`" path to a tab-separated file containing bulk expression profiles. The first column contains the gene names or ensemble 
      ids and the rest of the columns are bulk samples. See `./input/mixtures/pancreas/emtab_d` as an example.
   
   * `"org_prop_dir"` path to a tab-separated file containing the true cell type proportions in each bulk sample. This argument should be set to 
      `"null"` as true proportions are not known. However, for the purpose of method development you could use it. Read **Curriculum** section to see
      how this argumnet is used while running BEDwARS.
   
   * `"outdir"` root path to save the results. The output of BEDwARS is saved to `{outdir}/curric`, i.e., the name of the json file for curriculum 
      (`curric.json`) is appended to the outdir. After running the sampler (step 1 in **Running BEDwARS**) there will be a folder named
      `Models` in this directory which contains the checkpoints, i.e., state of the model and metropolis hasting for all variables. After step 2 
      of **Running BEDwARS**, there will be a folder named `aggr` containing the inferred parameters in this directory.
   
   </details>
   
</details>


<details>
<summary><h2>Curriculum</h2></summary>
   
<!--    Explain about each module and how the model is saved. This part should clarify why the inference is done
   using a stage name and a range of stage indices -->
   
   BEDwRAS follows a sequenece of instructions called curriculum given by `configs/curric.json`. Here we explain what each mode of operation in the
   instruction set does. `configs/curric.json` is the curriculum used for all the experiments in the paper with the exception of the number of chains
   being different for benchmarking (150 chains) and deconvolution performed for DPD deficiency (100 chains).
   
   The content of `configs/curric.json` is pasted below. It contains a sequence of four stages (instruction sets). Each stage has a name `"stage_name"`, 
   an operation `"stage_opr"` and a set of arguments required to perform the operation `"stage_args"`. 
   
   Briefly, the following sequence of stages do,
   1. Initialize the model and Metropolis Hastings sampler for all variables. 150 chains are initialized for sampling (L0). The initialized model and 
      metropolis hasting sampler are saved as `L0.pt`. 
   2. Perform the random walk for 50K steps. The scalings used for parameter updates get tunned in this period. This stage 
      is consiered as the tunning phase of the walker (L1). The random walk is performed in 10 mini stages of 5000 steps.
      At the end of each mini satge the model and metropolis hasting sampler states are saved (`L1-0.pt`, ..., `L1-9.pt`).
   3. Perform random walk for another 150K steps but without tunning (L1_train). The random walk is performed in 30 mini stages of 5000 steps.
      At the end of each mini satge the model and metropolis hasting sampler states are saved (`L1_train-0.pt`, ..., `L1_train-29.pt`).
   4. Sort the chains based on mean squared error computed between estimated and true bulk profiles subset to marker genes. Then subset the chains.
      The main purpose of this stage is to reduce the number of chains wisely to get speed up (L2). After subsetting, model and metropolis hasting 
      states are saved as `L2.pt`.
   5. Perform random walk for $180*5000= 800K$ steps with the remaining chains without tunning. You can decrease the number of steps in this stage 
      for speed up (L3). The random walk is performed in 180 mini stages of 5000 steps. At the end of each mini satge the model and metropolis 
      hasting sampler states are saved (`L3-0.pt`, ..., `L3-179.pt`). **The saved model states, i.e., sampled variable values, will be used in the 
      final inference of the variable values**.
      
   
   ```json
   [
     {
       "stage_name": "L0",
       "stage_oper": "init",
       "stage_args": {"rng_seeds": ["start_step", [12345, 200]],
                      "model_config": {"tune_interval": 50,
                                       "props_alpha_distr": "uniform",
                                       "alpha_lower": 0.0, "alpha_upper": 30.0,
                                       "sigma_b_distr": "halfcauchy","beta_b": 1.0,
                                       "sigma_x_distr": "halfcauchy","beta_x": 5.0,
                                       "eps": 1e-9,"w0_dir": null,"fixed_sigs": false,
                                       "divide_B": true,"chains": 150
                                       },
                      "chain_config": {"W": "default", "sigma_x": "prior", "sigma_b": "default", "alpha": "prior"}
                     }
     },{
       "stage_name": "L1",
       "stage_oper": "mcmc_repeat",
       "stage_args": {"init_stage_name": "L0", "mini_stage_repeats": 10,
                      "mini_stage_name_formatter":"{stage_name}-{mini_stage_idx}",
                      "mini_stage_args": {"steps": 5000, "after_tune": false,
                                          "get_stats": true, "log_acc_rate": false,
                                          "log_var_vals": false, "log_interval": 1000}
                      }
     },{
       "stage_name": "L1_train",
       "stage_oper": "mcmc_repeat",
       "stage_args": {"init_stage_name": "L1-9", "mini_stage_repeats": 30,
                      "mini_stage_name_formatter":"{stage_name}-{mini_stage_idx}",
                      "mini_stage_args": {"steps": 5000, "after_tune": true,
                                          "get_stats": true, "log_acc_rate": false,
                                          "log_var_vals": false, "log_interval": 1000}
                      }
     },{
       "stage_name": "L2",
       "stage_oper": "sort_subset",
       "stage_args": {"past_stage_name": "L1_train-29", "sort_criterion": "MSE_marker",
                      "use_summary":true, "is_transformed": false, "topk": 20,
                      "extra_dict": {"FCs": [3.0, 4.0, 5.0]},
                      "chain_inds_range": null
                      }
     },{
       "stage_name": "L3",
       "stage_oper": "mcmc_repeat",
       "stage_args": {"init_stage_name": "L2", "mini_stage_repeats": 180,
                      "mini_stage_name_formatter":"{stage_name}-{mini_stage_idx}",
                      "mini_stage_args": {"steps": 5000, "after_tune": true,
                                          "get_stats": true, "log_acc_rate": false,
                                          "log_var_vals": false, "log_interval": 1000}
                      }
     }
   ]
   ```
   
 + <details>
   <summary><strong>Description of the Stages</strong></summary>
   
   * `"stage_name"` each stage has to have a name given by this argument. It is important to choose a stage name as the checkpoints (state of model
      and Metropolis Hasting sampler) are saved with names containing the stage names. It is recommended to keep the stage names short.
   
   * `"stage_opr"` specifies the opertaion performed in the stage. This can take values `"init"`, `"mcmc_repeat"`, and `"sort_subset"`. In the `"init"`
      stage, model and chains for Metropolis Hasting sampler are initialized. Random walk happens in the `"mcmc_repeat"` stage and in the `"sort_subset"`
      stage, chains are sorted and subset based on a sepcific criterion.
   
   * `"stage_args"` contains the arguments required to carry out the operation in the stage.
   
   If you need to modify the curriculum please read the detailed description of arguments needed for each type of stage operation.
   
   </details>
   
 + <details>
   <summary><strong>Description of the Stage Arguments</strong></summary>
   
   The stage arguments are dependant on the stage operation. The stage arguments for each stage opertaion are explained below.
   + <details>
     <summary><strong>init</strong></summary>

      ```json
        {
          "stage_name": "L0",
          "stage_oper": "init",
          "stage_args": {"rng_seeds": ["start_step", [12345, 200]],
                         "model_config": {"tune_interval": 50,
                                          "props_alpha_distr": "uniform",
                                          "alpha_lower": 0.0, "alpha_upper": 30.0,
                                          "sigma_b_distr": "halfcauchy","beta_b": 1.0,
                                          "sigma_x_distr": "halfcauchy","beta_x": 5.0,
                                          "eps": 1e-9,"w0_dir": null,"fixed_sigs": false,
                                          "divide_B": true,"chains": 150
                                          },
                         "chain_config": {"W": "default", "sigma_x": "prior", "sigma_b": "default", "alpha": "prior"}
                        }
        }
      ```
      * `"rng_seeds"` the seeds for the random number generators (rng) used for model initialization and sampling in the chains. 
      The value is a list of two items. The first item `"start_step"` is the method for providing rng seeds and the second item
      contains the input required for the sepcified method. `"start_step"` means that the starting seed number and the step
      size for increasing the seed number should be provided. In this example, the starting seed is 12345 and the step size is 200.
      Therefore, 150 chains will be initialized with seeds $12345, 12345+200, 12345+2 \times 200, ..., 12345+149 \times 200$.
      Different seeds lead to different random initializations in the begining, if prior initialization is used for the variables,
      and also affect the random path taken by the walker.
      
      * `"model_config"` a dictionary that mostly contains the type of distributions e.g., uniform, used for the variables and the distribution
      parameters. It also includes the number of sampler chains ("`chains`") to run in parallel. You should not modify the default values for model
      configuration except for the number of chains. The number of chains depends on the available memory. Read **Run Time and Memory Requirement** 
      on this.
   
      * "`chain_config`" contains the mode of initialization for the variables of the model. You should not modify the mode of initialization 
      for the current version of BEDwARS.
      
      The stage containing the init operation saves the model and Metropolis Hasting sampler after initialization. So after running a stage 
      with init operation there will be a file named stage_name.pt, e.g., `L0.pt`, in the `Models` folder in the results directory.
 
     </details>
   
   + <details>
     <summary><strong>mcmc_repeat</strong></summary>

      ```json
      {
       "stage_name": "L1",
       "stage_oper": "mcmc_repeat",
       "stage_args": {"init_stage_name": "L0", "mini_stage_repeats": 10,
                      "mini_stage_name_formatter":"{stage_name}-{mini_stage_idx}",
                      "mini_stage_args": {"steps": 5000, "after_tune": false,
                                          "get_stats": true, "log_acc_rate": false,
                                          "log_var_vals": false, "log_interval": 1000}
                      }
      }
      ```
      * `"init_stage_name"` the stage from which the walker starts from. This can be the name of any file containing the model and 
      Metroppolis Hasting (MH) sampler state saved in it. In the example above, the model and MH sampler that were initialized and saved to 
      `L0.pt` will be loaded and pursued by the walker.
   
      * `"mini_stage_repeats"` the number of random walk modules (mini stages) to run. Each random walk module takes a certain number of  
      user-defined, e.g., 5000, steps which is provided by `"mini_stage_args"`. **After each module of random walk is completed the model and MH 
      sampler states are saved**.
   
      * `"mini_stage_forrmatter"` the forrmat to save the checkpoint, i.e., model and MH states, after each module of random walk is completed.
      Using the format above, the checkpoints are saved as `L1-0.pt`, `L1-1.pt`, ..., `L1-9.pt` as there are 10 mini stage repeats.
   
      * `"mini_stage_args"` a dictionary containing the following adjustable settings,
          1. `"steps"` the number of sampling/random walk steps.
          2. `"after_tune"` (boolean) whether the chain is in the tunning phase or not. This can be set to `true` or `false`.
          3. `"get_stats"` (boolean) in case ground truth signatures or proportions are provided (see **Configurations** on org_prop_dir and org_sig_dir), 
              setting this to `true` logs multiple statistics computed between the inferred signatures/proportions and their corresponding 
              ground truths during the random walk. In case the ground truths for signatures and proportions are not available setting this 
              option to `true` just prints out the statistics for the inferred bulk expression profiles. **It is recommended to set this to `false`
              if the ground truths are not available**. All the statistics are written to `status` file in the results directory.
          4. `"log_acc_rate"` (boolean) setting this to `true` takes logs of the acceptance rate of the proposals for each variable during tunning. 
              **This is only useful for algorithm development when the acceptance rate should be monitored so please set it to `false`**. The 
              logs are written to `status` file in the results directory.
          5. `"log_var_vals"` (boolean) logs the variable values. Set it to `false`.
          6. `"log_interval"` (integer) the logging interval in steps which is used for items 3,4, and 5.
         
         If you do not intend to make modifications to BEDwARS algorithm or explore its dynamics, please set `"get_stats"`, `"log_acc_rate"`, and
         `"log_var_vals"` to `false` in all stages with `"mcmc_repeat"` operation and do not log any statistics.
         If any of these items are set to `true` the logs will be stored in `status` file in the results directory.

     </details>
   
   
   + <details>
     <summary><strong>sort_subset</strong></summary>

      ```json
      {
       "stage_name": "L2",
       "stage_oper": "sort_subset",
       "stage_args": {"past_stage_name": "L1_train-29", "sort_criterion": "MSE_marker",
                      "use_summary":true, "is_transformed": false, "topk": 20,
                      "extra_dict": {"FCs": [3.0, 4.0, 5.0]},
                      "chain_inds_range": null
                      }
      }
      ```
     * `"past_stage_name"` the name of the checkpoint file, containing the model and metropolis hasting states, to be loaded. It can be set 
      to any checkpoint file name that exists. In this example it is set to `"L1_train-29"` because sort and subsetting should be performed 
      on the last state that was saved from the previous stage which was stage L1 with 30 mini stages/modules of random walk.
   
     * `"sort_criterion"` the criterion to sort the chains. It should be set to `"MSE_marker"` which uses the mean squared error between inferred
      and true bulk expression profiles restricted to marker genes.
   
     * `"extra_dict"` contains the criteria to pick the markers. A list of fold changes `[3.0, 4.0, 5.0]` can be passed. For each number X in the list
      the genes with X times higher expression in one of the cell types than the others are used as markers to compute the criterion and sort the chains.
      The union of top k chains picked using multiple marker sets will be used.
   
     * `"topk"` the number of chains picked after the sorting is applied.
   
     * `"use_summary"`, `"is_transformed"`, and `"chain_inds_range"` should not be changed.

     </details>

   
   </details>
   
</details>


<details>
<summary><h2>Curriculum Modification</h2></summary>
   
   * Change the number of chains running in parallel in case of limited memory. For this modification, you need to change the 
     value of `"chains"` in the `"stage_args"` of the `"L0"` stage with the `"init"` operation.
   * Change the run time by reducing the number of steps in the last stage `"L3"`. The current `./configs/curric.json` file has 
     180 mini stages (`"mini_stage_repeats"`) for `"L3"` stage. You can decrease this number for speed up. All of the paper results
     were generated with the model check points saved at `L3-59.pt`, ..., `L3-159.pt` **which is equivalent to setting `"mini_stage_repeats"`
     to 160**. Depending on the problem you could let the chains runs longer or not. However, it is best to stick with the setting 
     that was used for all the experiments in the paper.
   * If you do not intend to make modifications to BEDwARS algorithm or explore its dynamics, please set `"get_stats"`, `"log_acc_rate"`, and
     `"log_var_vals"` to `false` in all stages with `"mcmc_repeat"` operation and do not log any statistics. If any of these items are set to 
     `true` the logs will be stored in `status` file in the results directory.
   * Further modifications of curriculum can be done for method development.
   
   
</details>   


<!-- <details open>
<summary><h2>Format of input files</h2></summary>
   Tab-seperated input files for reference signatures and bulk expression profiles.
   Examples can be found at this and that for reference signature (genes by cell types) and bulk profiles (genes by samples), respectively.
   The first column has to be the gene symbol or ens ids.

</details> -->


<details>
<summary><h2>Run Time and Memory Requirement</h2></summary>

   Run time of BEDwARS depends on the computational complexity and the type of GPU used. 
   Deconvolving 100 psedu-bulk samples of Segerstolpe T2D (`./input/mixtures/pancreas/emtab_d`) 
   using Baron cell type signatures (`./input/signatures/pancreas/Baron/Baron`) and `./configs/curric.json`
   as the curriculum takes ~100 ms per step (~16 hrs in total) on Tesla V100 GPU.

   The memory requierement depends mainly on the number of genes and chains. For the same experiment with
   ~8000 genes, 6 cell types, 100 pseudo-bulk samples, and 150 chains we used 16 Gb of memory. To find the
   memory requirement for your experiments do the following calculations,
   
   Assuming that
   * $G$ is the number of genes,
   * $N$ is the number of bulk samples,
   * $C$ is the number of cell types, and
   * $K$ is the number of sampler chains
   
   The memory consumption is roughly $4 \times (G + N) \times C \times K$ bytes, where 32-bit floating point
   numbers are used.

   
   
</details>


## References
* The bioRxiv link to the paper:
  * PDF link: [https://www.biorxiv.org/content/10.1101/2022.10.25.513800v1.full.pdf)
  * Web-page link: [https://www.biorxiv.org/content/10.1101/2022.10.25.513800v1)

* Here is the bibtex citation entry for our work:
```
@article{ghaffari2022robust,
  title={A Robust Bayesian Approach to Bulk Gene Expression Deconvolution with Noisy Reference Signatures},
  author={Ghaffari, Saba and Bouchonville, Kelly J and Saleh, Ehsan and Schmidt, Remington E and Offer, Steven M and Sinha, Saurabh},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```
