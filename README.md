# bns_mass_distribution
This repository accompanies https://arxiv.org/abs/2111.13619, and demonstrates how to reproduce the results in that paper where there is no spin prior mismatch, so that the resulting mass distribution should be unbiased. 

## preliminaries
You will need to install [bilby](https://lscsoft.docs.ligo.org/bilby/installation.html), [pymultinest](https://anaconda.org/conda-forge/pymultinest), [nestle](http://kylebarbary.com/nestle/) and [gwpopulation](https://colmtalbot.github.io/gwpopulation/). The first two can be installed using conda, and the third using `pip`. 
`bilby` is the inference library that will facilitate both the compact binary and hierarchical parameter estimation, `pymultinest` is the sampler that will be used for the initial PE, `gwpopulation` includes models for conducting hierarchical inference for compact binaries, and `nestle` is the sampler used for the hierarchical inference.

## running the code
The first step is to generate the injections used to evaluate the Monte Carlo integral for the selection function and the 100 simulated "detected" events for which we will run individual parameter estimation.
This is done by calling
```
python generate_injections_O3.py NUMBER
```
where `NUMBER` is the total number of found injections you want to use _plus_ the 100 detected events. In the paper, we used `NUMBER=40100`. 
This will produce a file with injections called `data/injection_list_O3_all_flat_med_spin.dat`, and 100 `json` files in the `data` directory containing the injection parameter dictionaries for each of the detected events.

Next, we run parameter estimation for each of the 100 detected events. We use the reduced order quadrature implementation of the IMRPhenomPv2 waveform to reduce the computational cost, so first we need to compute the ROQ weights.
If not running this on a LIGO Data Grid machine where the ROQ bases are already available, you will need to [download the 128s basis](https://git.ligo.org/lscsoft/ROQ_data/-/tree/master/IMRPhenomPv2/128s) and change the marked file path in the two scripts below.
The weight calculation is split into a separate script because it is quite memory-intensive (~64GB), while the actual sampling requires much less memory (~4 GB).
The weights are computed for each individual event using:
```
python roq_compute_weights_O3.py NUMBER data/
```
where `NUMBER` is the number of the "detected" event being analyzed, and `data/` tells the code to write the output to the `data/` directory. Once this step is complete, you can run parameter estimation for each of these events by calling:
```
python roq_run_sampler_mn_O3.py NUMBER data/
```
This will produce a file called `roq_O3_result.json` and a corresponding corner plot. The raw output of the `pymultinest` run will also be saved to a directory called `data/pm_roq_O3/`, but you won't need anything from that directory.
Once you have done this for each of the 100 events (or fewer, if you don't want to analyze that many), you can run hierarchical inference to infer the values of the hyper-parameters describing the mass distribution. 
This is done using:
```
python hyper_pe_O3.py NUMBER
```
where this time `NUMBER` is the number of the last event you wanted to be included in the analysis. We used `NUMBER=99` in the paper. The output of this step will be a similar result file and corner plot called `data/hyper_noq_medSpinInj_medSpin_reflected_result.json` with the posterior samples for the mass hyper-parameters.

## introducing a spin prior mismatch
If you wanted to modify the code to instead introduce a spin prior mismatch as is explored in the paper, you will need to change the `.prior` files in the `data/` directory.
To change the maximum spin magnitude of the _injected_ and _detected_ events, change the lines for `chi_1` and `chi_2` in `data/binary_neutron_star_injections_O3.prior`.
To change the prior applied during parameter estimation for the individual events, change the lines for `chi_1` and `chi_2` in `data/binary_neutron_stars_O3.prior`.
In principle, the spin distribution used for the injections (not the detected events) should match the one applied during parameter estimation, since you should use the same assumptions about the universe in both places. 
In this case it's often easiest to generate the injections from a wider spin prior and then reweight them to match your desired PE prior during the hiearchical inference stage. 
The three auxiliary functions at the beginning of the `hyper_pe_O3.py` script are useful for this purpose, though we leave a full demonstration of this reweighting as an exercise to the reader.
