#!/usr/bin/env python
"""
author: S. Biscoveanu, Jan 12 2022
Example of how to use the Reduced Order Quadrature method (see Smith et al.,
(2016) Phys. Rev. D 94, 044031) for a Binary Black hole simulated signal in
Gaussian noise.

This requires files specifying the appropriate basis weights.
These aren't shipped with Bilby, but are available on LDG clusters and
from the public repository https://git.ligo.org/lscsoft/ROQ_data.
"""
from __future__ import division, print_function
import numpy as np
import json
import bilby
import sys

class UniformReflected(bilby.core.prior.analytical.Uniform):
    def rescale(self, val):
        u = 2 * np.minimum(val, 1 - val)
        return super().rescale(u)

def main(number, outdir):
    label = 'roq_O3_'+str(number)
    np.random.seed(3*number)
    injection_parameters = json.load(open(outdir+'/'+label+'_inj.json','r'))

    # Rescale the 128s basis
    if injection_parameters['chirp_mass'] >= 1.5:
        flow = 20.
    else:
        flow = 32.
    rescale_factor = flow/20
    mc_min = 1.420599/rescale_factor
    mc_max = 2.602169/rescale_factor

    # Load in the frequency nodes for the ROQ. Note you will need to
    # adjust the filenames here to the correct paths on your machine
    roq_matrix_directory = '/home/cbc/ROQ_data/IMRPhenomPv2/128s/'
    freq_nodes_linear = np.load(roq_matrix_directory+"fnodes_linear.npy")*rescale_factor
    freq_nodes_quadratic = np.load(roq_matrix_directory+"fnodes_quadratic.npy")*rescale_factor
    params = np.genfromtxt(roq_matrix_directory+"params.dat", names=True)
    params['flow'] *= rescale_factor
    params['fhigh'] *= rescale_factor
    params['seglen'] /= rescale_factor
    params['chirpmassmin'] /= rescale_factor
    params['chirpmassmax'] /= rescale_factor
    params['compmin'] /= rescale_factor

    duration = 128. / rescale_factor 
    sampling_frequency = 8192. * rescale_factor

    # Set up the prior
    tc = injection_parameters['geocent_time']
    priors = bilby.gw.prior.BBHPriorDict(filename=outdir+'/binary_neutron_stars_O3.prior')
    priors['mass_ratio'] = UniformReflected(name='mass_ratio', minimum=0.125, maximum=1.)
    priors['mass_1'].minimum = params['compmin']
    priors['mass_2'].minimum = params['compmin']
    priors['chirp_mass'].minimum = max(mc_min, injection_parameters['chirp_mass']-0.1)
    priors['chirp_mass'].maximum = min(mc_max, injection_parameters['chirp_mass']+0.1)
    priors['geocent_time'] = bilby.core.prior.Uniform(tc - 0.1, tc + 0.1, latex_label='$t_c$', unit='s')
    distance_marginalization=True
    
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    for ifo in ifos:
        ifo.minimum_frequency = flow
        ifo.maximum_frequency = sampling_frequency/2.
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=outdir+'/aligo_O3actual_{}.txt'.format(ifo.name))
        freqs, real, imag = np.loadtxt('{}/{}_{}_frequency_domain_data.dat'.format(outdir, ifo.name, label), unpack=True, skiprows=1)
        strain = real + 1j*imag
        ifo.set_strain_data_from_frequency_domain_strain(strain,
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=injection_parameters['geocent_time'] +2 - duration)

    # make ROQ waveform generator
    search_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.binary_black_hole_roq,
        waveform_arguments=dict(frequency_nodes_linear=freq_nodes_linear,
                                frequency_nodes_quadratic=freq_nodes_quadratic,
                                reference_frequency=100., minimum_frequency=flow,
                                approximant='IMRPhenomPv2'),
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)

    # load the weights from the file
    likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient(
        interferometers=ifos, waveform_generator=search_waveform_generator,
        weights=outdir+'/'+label+'_weights.json.npz', priors=priors, roq_params=params,
        distance_marginalization=distance_marginalization, phase_marginalization=True)

    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler='pymultinest', npoints=1000, sampling_efficiency=0.1,
        injection_parameters=injection_parameters, outdir=outdir, label=label, resume=True,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters)

    # Make a corner plot.
    parameters=[u'psi', u'chirp_mass', u'ra', u'dec', u'mass_ratio', u'theta_jn', u'geocent_time', u'luminosity_distance', u'chi_1', u'chi_2']
    parameter_dict = dict()
    for param in parameters:
        parameter_dict[param] = result.injection_parameters[param]
    result.plot_corner(parameters=parameter_dict)

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print("Usage is: \n > python roq_run_sampler_mn_O3.py NUMBER OUTDIR")
        print("NUMBER is the run number")
        print("OUTDIR is where the output will be written")
    else:
        main(int(sys.argv[1]), sys.argv[2])
