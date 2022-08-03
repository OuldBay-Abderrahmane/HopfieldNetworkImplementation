import sys
import doctest
import algorithms as alg
import numpy as np


def test_class_hebbian_weights_speed(benchmark):
    patterns = alg.generate_patterns(50, 2500)
    test = alg.HopfieldNetwork(patterns, rule='hebbian')
    benchmark.pedantic(test.hebbian_weights, patterns)


def test_class_storkey_weights_speed(benchmark):
    patterns = alg.generate_patterns(50, 2500)
    test = alg.HopfieldNetwork(patterns, rule='hebbian')
    benchmark(test.storkey_weights, patterns)


def test_update_speed(benchmark):
    patterns = alg.generate_patterns(50, 2500)
    perturbed = alg.perturb_pattern(patterns[0], 1000)
    test = alg.HopfieldNetwork(patterns, rule='hebbian')

    def update_speed(pattern):
        for i in range(100):
            test.update(pattern)

    benchmark(update_speed, perturbed)


def test_update_async_speed(benchmark):
    patterns = alg.generate_patterns(50, 2500)
    test = alg.HopfieldNetwork(patterns, rule='hebbian')
    perturbed = alg.perturb_pattern(patterns[0], 1000)

    def update_async_speed(pattern):
        for i in range(100):
            test.update_async(pattern)

    benchmark(update_async_speed, perturbed)


def test_energy_speed(benchmark):
    patterns = alg.generate_patterns(50, 2500)
    test_weights = alg.HopfieldNetwork(patterns, rule='hebbian')
    test = alg.DataSaver()
    benchmark(test.compute_energy,
              patterns[0], test_weights.hebbian_weights(patterns))


def test_dynamics_speed(benchmark):
    patterns = alg.generate_patterns(80, 1000)
    perturbed = alg.perturb_pattern(patterns[0], 10)
    test = alg.HopfieldNetwork(patterns, rule='hebbian')

    benchmark(test.dynamics, perturbed, test.hebbian_weights(patterns), 20)
