#!/usr/bin/env python
"""
author: S. Biscoveanu, Jan. 12 2022
Generate a set of injections to use when evaluating the selection bias
in the hierarchical inference likelihood.
"""
from __future__ import division, print_function
import numpy as np
import json
import bilby
import sys
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy import units
from pandas import DataFrame


def main(number):
    bilby.core.utils.setup_logger(log_level='WARNING')
    # define source frame total mass prior
    mmin = 2.3
    alpha = 2.5
    mmax = 4.3
    deltam = 0.4

    # load prior distributions
    np.random.seed(3 * number)
    inj_priors = bilby.gw.prior.BBHPriorDict(
        filename="./data/binary_neutron_star_injections_O3.prior"
    )
    q_arr = np.linspace(0.4, 1, 500)
    q_unif = bilby.core.prior.Uniform(0.4, 1, name="mass_ratio")
    pq = 1.0 / 2 * q_unif.prob(q_arr) + 1.0 / 2 * inj_priors["mass_ratio"].prob(q_arr)
    inj_priors["mass_ratio"] = bilby.core.prior.Interped(
        q_arr, pq, minimum=0.4, maximum=1.0, name="mass_ratio"
    )
    inj_list = []
    i = 0

    flow = 20.0
    duration = 128.0
    sampling_frequency = 8192.0

    # make waveform generator
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=dict(
            reference_frequency=100.0,
            minimum_frequency=flow,
            approximant="IMRPhenomPv2",
        ),
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    )

    # setup interferometers
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in ifos:
        ifo.minimum_frequency = flow
        ifo.maximum_frequency = sampling_frequency / 2.0
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            asd_file="./data/aligo_O3actual_{}.txt".format(ifo.name)
        )

    # generate injections
    while i < number:
        print(i)
        tc = 1126259642.413 + 128 * i
        injection_parameters = inj_priors.sample()
        injection_parameters["geocent_time"] = tc
        z = z_at_value(
            cosmo.luminosity_distance,
            injection_parameters["luminosity_distance"] * units.Mpc,
        )
        injection_parameters["total_mass"] = injection_parameters[
            "total_mass_source"
        ] * (1 + z)
        injection_parameters["mass_1"] = injection_parameters["total_mass"] / (
            1 + injection_parameters["mass_ratio"]
        )
        injection_parameters["mass_2"] = (
            injection_parameters["mass_1"] * injection_parameters["mass_ratio"]
        )
        injection_parameters["chirp_mass"] = (
            injection_parameters["mass_1"] * injection_parameters["mass_2"]
        ) ** (3.0 / 5) / (
            injection_parameters["mass_1"] + injection_parameters["mass_2"]
        ) ** (
            1.0 / 5
        )

        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=(injection_parameters["geocent_time"] + 2 - duration),
        )
        ifos.inject_signal(
            parameters=injection_parameters, waveform_generator=waveform_generator
        )
        rho_opt_2 = 0
        for ifo in ifos:
            rho_opt_2 += ifo.meta_data["optimal_SNR"] ** 2
        if i >= 100:
            # these are the injections
            injection_parameters["network_optimal_snr"] = np.sqrt(rho_opt_2)
            inj_list.append(injection_parameters)
        if np.sqrt(rho_opt_2) >= 9:
            if i < 100:
                # these are the events
                json.dump(injection_parameters, open('./data/roq_O3_{}_inj.json'.format(i),'w'))
            i += 1

    inj_df = DataFrame(inj_list)
    inj_df.to_csv("./data/injection_list_O3_all_flat_med_spin.dat", index=False, sep="\t")
    print("Done")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage is: \n > python generate_injections_O3.py NUMBER")
        print("NUMBER is the number of found injections including 100 events")
    else:
        main(int(sys.argv[1]))
