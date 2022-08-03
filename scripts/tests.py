import algorithms as alg
import numpy as np


def test_storkey_weights():
    assert np.allclose(alg.storkey_weights(np.array([[1., 1, -1., -1.],
                                                     [1., 1., -1, 1.], [-1., 1, -1., 1.]])), np.array([[1.125,  0.25, -0.25, -0.5],
                                                                                                       [0.25,  0.625, -1.,  0.25],
                                                                                                       [-0.25, -1.,
                                                                                                           0.625, -0.25],
                                                                                                       [-0.5,  0.25, -0.25,  1.125]]))


def test_hebbian_weights():
    assert np.allclose(alg.hebbian_weights(np.array([[1., 1, -1., -1.],
                                                     [1., 1., -1, 1.], [-1., 1, -1., 1.]])), np.array([[0., 1/3, -1/3,  -1/3],
                                                                                                       [1/3,  0., -1.,  1/3],
                                                                                                       [-1/3, -1.,
                                                                                                           0.,  -1/3],
                                                                                                       [-1/3, 1/3, -1/3, 0.]]))
    assert np.shape(alg.hebbian_weights(
        alg.generate_patterns(3, 50))) == (50, 50)
    assert np.allclose(np.diag(alg.hebbian_weights(alg.generate_patterns(
        3, 10))), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))


def test_dynamics():
    patterns = alg.generate_patterns(80, 1000)
    perturbed = alg.perturb_pattern(patterns[0], 10)
    fund = alg.dynamics(perturbed, alg.hebbian_weights(patterns), 20)
    assert np.allclose(fund[-1], patterns[0])


def test_dynamics_async():
    patterns = alg.generate_patterns(80, 1000)
    perturbed = alg.perturb_pattern(patterns[0], 10)
    fund = alg.dynamics_async(
        perturbed, alg.hebbian_weights(patterns), 100000, 10000)
    assert np.array_equal(fund[-1], patterns[0])


def test_hebbian_weights_speed(benchmark):
    patterns = alg.generate_patterns(50, 2500)
    benchmark(alg.hebbian_weights, patterns)


def test_storkey_weights_speed(benchmark):
    patterns = alg.generate_patterns(50, 2500)
    benchmark(alg.storkey_weights, patterns)


def test_update_speed(benchmark):
    def update_speed(pattern, weight):
        for i in range(100):
            alg.update(pattern, weight)
    patterns = alg.generate_patterns(50, 2500)
    perturbed = alg.perturb_pattern(patterns[0], 1000)
    benchmark(update_speed, perturbed, alg.hebbian_weights(patterns))


def test_update_async_speed(benchmark):
    def update_async_speed(pattern, weight):
        for i in range(100):
            alg.update_async(pattern, weight)

    patterns = alg.generate_patterns(50, 2500)
    perturbed = alg.perturb_pattern(patterns[0], 1000)
    benchmark(update_async_speed, perturbed, alg.hebbian_weights(patterns))


def test_energy_speed(benchmark):
    patterns = alg.generate_patterns(50, 2500)
    weights = alg.hebbian_weights(patterns)
    benchmark(alg.energy, patterns[0], weights)
