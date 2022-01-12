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

def main(number, outdir):
    label = 'roq_O3_'+str(number)
    np.random.seed(3*number)

    # Set up the injection
    print('Loading injection')
    injection_parameters = json.load(open(outdir+'/'+label+'_inj.json', 'r'))
    tc = injection_parameters['geocent_time']

    # Rescale the 128s basis
    print('Setting up the ROQ')
    if injection_parameters['chirp_mass'] >= 1.5:
        flow = 20.
    else:
        flow = 32.
    rescale_factor = flow/20
    mc_min = 1.420599/rescale_factor
    mc_max = 2.602169/rescale_factor

    # Load in the pieces for the linear part of the ROQ. Note you will need to
    # adjust the filenames here to the correct paths on your machine
    roq_matrix_directory = '/home/cbc/ROQ_data/IMRPhenomPv2/128s/'
    basis_matrix_linear = np.load(roq_matrix_directory+"B_linear.npy").T
    freq_nodes_linear = np.load(roq_matrix_directory+"fnodes_linear.npy")*rescale_factor

    # Load in the pieces for the quadratic part of the ROQ
    basis_matrix_quadratic = np.load(roq_matrix_directory+"B_quadratic.npy").T
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

    # setup injection waveform generator
    print('Setting up waveform generator and interferometers')
    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                              reference_frequency=100., minimum_frequency=flow)

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)

    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    for ifo in ifos:
        ifo.minimum_frequency = flow
        ifo.maximum_frequency = sampling_frequency/2.
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=outdir+'/aligo_O3actual_{}.txt'.format(ifo.name))
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=(injection_parameters['geocent_time'] + 2 - duration))
    ifos.inject_signal(waveform_generator=waveform_generator,
                       parameters=injection_parameters)
    for ifo in ifos: ifo.save_data(outdir, label)

    # make ROQ waveform generator
    search_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.binary_black_hole_roq,
        waveform_arguments=dict(frequency_nodes_linear=freq_nodes_linear,
                                frequency_nodes_quadratic=freq_nodes_quadratic,
                                reference_frequency=100., minimum_frequency=flow,
                                approximant='IMRPhenomPv2'),
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)

    # setup sampling prior
    print('Creating prior')
    priors = bilby.gw.prior.BBHPriorDict(filename=outdir+'/binary_neutron_stars_O3.prior')
    priors['mass_1'].minimum = params['compmin']
    priors['mass_2'].minimum = params['compmin']
    priors['chirp_mass'].minimum = max(mc_min, injection_parameters['chirp_mass']-0.1)
    priors['chirp_mass'].maximum = min(mc_max, injection_parameters['chirp_mass']+0.1)
    priors['geocent_time'] = bilby.core.prior.Uniform(tc - 0.1, tc + 0.1, latex_label='$t_c$', unit='s')

    print('Creating likelihood and ROQ weights')
    likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient(
        interferometers=ifos, waveform_generator=search_waveform_generator,
        linear_matrix=basis_matrix_linear, quadratic_matrix=basis_matrix_quadratic, roq_params=params,
        priors=priors, distance_marginalization=True, phase_marginalization=True)

    # write the weights to file so they can be loaded multiple times
    likelihood.save_weights(outdir+'/'+label+'_weights.json')

    # remove the basis matrices as these are big for longer bases
    del basis_matrix_linear, basis_matrix_quadratic

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print("Usage is: \n > python roq_compute_weights.py NUMBER OUTDIR")
        print("NUMBER is the run number")
        print("OUTDIR is where the output will be written")
    else:
        main(int(sys.argv[1]), sys.argv[2])
