import bilby
import h5py
import json
import gwpopulation as gwpop
import numpy as np
import sys
from gwpopulation.cupy_utils import xp
import pdb
from copy import copy
from pandas import DataFrame, read_csv
from bilby.hyper.model import Model

def truncnorm_q(dataset, mu, sigma, high, low):
    return gwpop.utils.truncnorm(dataset['mass_ratio'], mu, sigma, high, low)

# spin models
def med_spin(dataset): 
    pe_prior = bilby.gw.prior.AlignedSpin(a_prior=bilby.core.prior.Uniform(0, 0.4), z_prior=bilby.core.prior.Uniform(-1, 1))
    return pe_prior.prob(dataset['chi_1'])*pe_prior.prob(dataset['chi_2'])

def high_spin(dataset):
    pop_prior = bilby.gw.prior.AlignedSpin(a_prior=bilby.core.prior.Uniform(0, 0.8), z_prior=bilby.core.prior.Uniform(-1, 1))
    return pop_prior.prob(dataset['chi_1'])*pop_prior.prob(dataset['chi_2'])

def low_spin(dataset):
    pop_prior = bilby.gw.prior.AlignedSpin(a_prior=bilby.core.prior.Uniform(0, 0.05), z_prior=bilby.core.prior.Uniform(-1, 1))
    return pop_prior.prob(dataset['chi_1'])*pop_prior.prob(dataset['chi_2'])

class SinglePeakSmoothedMassDistribution(gwpop.models.mass.BaseSmoothedMassDistribution):
    def __init__(self):
        super(SinglePeakSmoothedMassDistribution, self).__init__()
        self.m1s = xp.linspace(2, 5, 500)
        self.dm = self.m1s[1] - self.m1s[0]

    def __call__(self, dataset, alpha, mmin, mmax, lam, mpp, sigpp, delta_m):
        """
        Powerlaw + peak model for two-dimensional mass distribution with low
        mass smoothing.
        https://arxiv.org/abs/1801.02699 Eq. (11) (T&T18)
        Parameters
        ----------
        dataset: dict
            Dictionary of numpy arrays for 'total_mass_source' and 'mass_ratio'.
        alpha: float
            Powerlaw exponent for more massive black hole.
        mmin: float
            Minimum black hole mass.
        mmax: float
            Maximum mass in the powerlaw distributed component.
        lam: float
            Fraction of black holes in the Gaussian component.
        mpp: float
            Mean of the Gaussian component.
        sigpp: float
            Standard deviation fo the Gaussian component.
        delta_m: float
            Rise length of the low end of the mass distribution.
        """
        p_mtot = self.p_mtot(
            dataset,
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
            delta_m=delta_m,
        )
        return p_mtot

    def p_mtot(self, dataset, alpha, mmin, mmax, lam, mpp, sigpp, delta_m):
        p_m = gwpop.models.mass.two_component_single(
            dataset["total_mass_source"],
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        p_m *= self.smoothing(dataset["total_mass_source"], mmin=mmin, mmax=mmax, delta_m=delta_m)
        norm = self.norm_p_mtot(
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
            delta_m=delta_m,
        )
        return p_m / norm

    def norm_p_mtot(self, alpha, mmin, mmax, lam, mpp, sigpp, delta_m):
        """Calculate the normalisation factor for the primary mass"""
        if delta_m == 0.0:
            return 1
        p_m = gwpop.models.mass.two_component_single(
            self.m1s, alpha=alpha, mmin=mmin, mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp
        )
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=mmax, delta_m=delta_m)

        norm = gwpop.cupy_utils.trapz(p_m, self.m1s)
        return norm

def main(number):
    print(xp)
    # load the PE posteriors
    outdir = './data/'
    posteriors = []
    for i in range(number+1):
        try: 
            result = json.load(open(outdir+'roq_O3_'+str(i)+'_result.json', 'r'))
        except IOError:
            print('Skipping {}'.format(i))
            continue

        result_df = DataFrame.from_dict(result['posterior']['content'])
        q = result_df['mass_ratio']
        result_df['prior'] = ((q / (1 + q)**2)**(3./5)) * (1 + result_df['redshift'])
        posteriors.append(result_df[['total_mass_source', 'mass_ratio', 'prior']])
    
    # population priors
    mtot_dist = SinglePeakSmoothedMassDistribution()
    model = Model([mtot_dist, truncnorm_q])
    pdet_model = Model([SinglePeakSmoothedMassDistribution()])

    # selection effects
    o3a_injections = read_csv(outdir+'/injection_list_O3_all_flat_med_spin.dat', sep='\t', header=0)
    total_generated = len(o3a_injections)
    o3a_found = copy(o3a_injections[o3a_injections["network_optimal_snr"] >= 9])
    print('Found injections: {}\n'.format(len(o3a_found)))
    truths = {'mmin': 2.3, 'mmax': 4.3, 'alpha': 2.5, 'delta_m': 0.4, 'mu':1., 'sigma':0.1}
    truths_full = {'lam': 0, 'mpp': 4, 'sigpp': 2, 'low': 0.4, 'high': 1., **truths}
    
    # the injection prior is flat in Mtot, ignore q since not included in pdet_model
    pdet_model.parameters = copy(truths_full)
    o3a_found = {key: xp.asarray(o3a_found[key]) for key in o3a_found}
    inj_priors = bilby.gw.prior.BBHPriorDict(filename='/home/sylvia.biscoveanu/populations/bns_accretion/binary_neutron_star_injections_O3.prior')
    # this is overkill since the prior is flat, but included for demonstration
    o3a_found['prior'] = inj_priors['total_mass_source'].prob(o3a_found['total_mass_source'])
    o3a_found['total_generated'] = total_generated
    N_EVENTS = len(posteriors)
    print(N_EVENTS)

    # code below adapted from gwpopulation to ignore overall rate
    class ResamplingVT(object):
        def __init__(self, model, data):
            self.data = data
            if isinstance(model, list):
                model = Model(model)
            elif not isinstance(model, Model):
                model = Model([model])
            self.model = model

        def __call__(self, parameters):
            """
            Compute the expected number of detections given a set of injections.

            This should be implemented as in https://arxiv.org/abs/1904.10879

            If 4 * n_events < n_effective we return np.inf so that the sample
            is rejected.

            Parameters
            ----------
            parameters: dict
                The population parameters
            """
            mu, var = self.detection_efficiency(parameters)
            if var == 0:
                return np.inf
            n_effective = mu ** 2 / var
            if n_effective < 4 * N_EVENTS:
                print("\n\nUndersampled part of hyperparameter space\n{}".format(parameters))
                return np.inf
            vt_factor = mu / np.exp((3 + N_EVENTS) / 2 / n_effective)
            return vt_factor

        def compute_weights(self, parameters):
            self.model.parameters.update(parameters)
            return xp.nan_to_num(self.model.prob(self.data) / self.data["prior"])
        
        def draw_weighted_samples(self, parameters, keys=None):
            if keys is None:
                keys = list()
            weights = self.compute_weights(parameters)
            keep = weights > xp.random.uniform(0, xp.max(weights), len(weights))
            return xp.asarray([self.data[key][keep] for key in keys])

        def detection_efficiency(self, parameters):
            weights = self.compute_weights(parameters)
            mu = float(xp.sum(weights / self.data["total_generated"]))
            var = float(
                xp.sum((weights / self.data["total_generated"]) ** 2)
                - mu ** 2 / self.data["total_generated"]
            )
            return mu, var

    resampling_vt = ResamplingVT(model=pdet_model, data=o3a_found)
    # likelihood
    fast_likelihood = gwpop.hyperpe.HyperparameterLikelihood(
	posteriors=posteriors, hyper_prior=model,
	selection_function=resampling_vt
        )

    # define the hyper-priors
    hyper_priors = bilby.core.prior.PriorDict()
    hyper_priors['alpha'] = bilby.core.prior.Uniform(minimum=0, maximum=4, latex_label='$\\alpha$')
    hyper_priors['mmin'] = bilby.core.prior.Uniform(minimum=2.0, maximum=3.0, latex_label='$m_{\\min}$')
    hyper_priors['mmax'] = bilby.core.prior.Uniform(minimum=3.2, maximum=5.0, latex_label='$m_{\\max}$')
    hyper_priors['delta_m'] = bilby.core.prior.Uniform(minimum=0, maximum=1, latex_label='$\\delta_{m}$')
    hyper_priors['lam'] = 0
    hyper_priors['mpp'] = 4
    hyper_priors['sigpp'] = 2
    hyper_priors['mu'] = bilby.core.prior.Uniform(minimum=0.4, maximum=1.0, latex_label='$\\mu_{q}$')
    hyper_priors['sigma'] = bilby.core.prior.Uniform(minimum=0.01, maximum=0.5, latex_label='$\\sigma_{q}$')
    hyper_priors['high'] = 1
    hyper_priors['low'] = 0.4

    # test likelihood
    prior_sample = hyper_priors.sample()
    fast_likelihood.parameters = prior_sample
    print(fast_likelihood.log_likelihood_ratio())

    # run sampler
    hyper_result = bilby.run_sampler(likelihood=fast_likelihood, priors=hyper_priors, sampler='nestle',
                                     nlive=1000, label='hyper_noq_medSpinInj_medSpin_reflected_'+str(number), outdir=outdir)
    hyper_result.plot_corner(truths=truths)

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print("Usage is: \n > python hyper_pe_O3.py OUTDIR")
        print("NUMBER is the number of the last individual event to be included in the run")
    else:
        main(int(sys.argv[1]))

