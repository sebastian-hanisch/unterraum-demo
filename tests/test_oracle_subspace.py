"""Kern der Unterraumverfahren: Mini-Instanz von Hand, scipy-Orakel, rauschfreie Wiederfindung, Linienzahl gegen eine unabhängige Schleife, Cramér-Rao-Schranke."""

import math

import numpy as np
import pytest
from scipy import linalg

import ure_subspace as U

S2 = math.sqrt(0.5)


def test_snapshots_and_covariance_of_the_mini_instance_by_hand(mini_x):
    s = U.snapshots(mini_x, 3)
    assert s.tolist() == [[1, 0, -1], [0, -1, 0], [-1, 0, 1], [0, 1, 0]]
    r = U.covariance(mini_x, 3)
    assert r == pytest.approx(np.array([[0.5, 0.0, -0.5], [0.0, 0.5, 0.0], [-0.5, 0.0, 0.5]]), abs=1e-14)
    assert U.covariance(mini_x, 3, forward_backward=False) == pytest.approx(r, abs=1e-14)         # schon vorwärts-rückwärts-symmetrisch


def test_eigendecomposition_of_the_mini_instance_is_sorted_downwards(mini_x):
    w, v = U.eigendecomposition(U.covariance(mini_x, 3))
    assert w == pytest.approx([1.0, 0.5, 0.0], abs=1e-12)
    assert abs(v[:, 0]) == pytest.approx([S2, 0.0, S2], abs=1e-12) and abs(v[:, 1]) == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert abs(v[:, 2]) == pytest.approx([S2, 0.0, S2], abs=1e-12)


def test_music_pseudospectrum_has_a_pole_at_the_line_of_the_mini_instance(mini_x):
    w, v = U.eigendecomposition(U.covariance(mini_x, 3))
    # Rauschvektor (1, 0, 1)/sqrt 2: a(f) = (1, e^{-i 2 pi f}, e^{-i 4 pi f}); bei f = 1/4 ist a = (1, -i, -1), senkrecht dazu: Nenner 0
    f = np.array([0.1, 0.25 - 1e-9, 0.4])
    p = U.music_spectrum(v, 2, f)
    assert p[1] > 1e12 and p[0] < 10 and p[2] < 10
    assert U.music_spectrum(v, 2, np.array([0.0])) == pytest.approx([1.0 / 2.0], abs=1e-12)        # f = 0: a = (1, 1, 1), ||E_n^T a||^2 = (2 / sqrt 2)^2 / ... = 2
    with pytest.raises(ValueError):
        U.music_spectrum(v, 0, f)
    with pytest.raises(ValueError):
        U.music_spectrum(v, 3, f)


def test_esprit_and_root_music_of_the_mini_instance(mini_x):
    w, v = U.eigendecomposition(U.covariance(mini_x, 3))
    assert U.esprit(v, 2) == pytest.approx([0.25], abs=1e-9)                                       # Phi = [[0, sqrt 2], [-1/sqrt 2, 0]], Eigenwerte +-i
    assert U.root_music(v, 2) == pytest.approx([0.25], abs=1e-9)
    assert U.music_frequencies(v, 2, 1, np.arange(1, 64) / 128.0) == pytest.approx([0.25], abs=1e-4)


def test_estimate_count_of_two_large_eigenvalues_and_a_noise_floor():
    w = np.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0])                                                 # ab k = 2 sind die übrigen gleich: log(g / a) = 0
    assert U.estimate_count(w, 100, "MDL") == 2 and U.estimate_count(w, 100, "AIC") == 2
    flat = np.ones(8)                                                                              # nur Rauschen: kein Signal
    assert U.estimate_count(flat, 100, "MDL") == 0 and U.estimate_count(flat, 100, "AIC") == 0


def test_covariance_validation_and_forward_backward_symmetry():
    with pytest.raises(ValueError):
        U.snapshots(np.ones(5), 1)
    with pytest.raises(ValueError):
        U.snapshots(np.ones(5), 6)
    x = np.random.default_rng(3).normal(size=50)
    r = U.covariance(x, 12)
    assert r == pytest.approx(r.T, abs=1e-14) and r == pytest.approx(r[::-1, ::-1], abs=1e-14)
    r0 = U.covariance(x, 12, forward_backward=False)
    assert r == pytest.approx(0.5 * (r0 + r0[::-1, ::-1]), abs=1e-14) and not np.allclose(r0, r0[::-1, ::-1])


def test_covariance_equals_the_double_sum_and_eigh_equals_scipy():
    x = np.random.default_rng(5).normal(size=40)
    m = 6
    n_snap = len(x) - m + 1
    ref = np.array([[sum(x[t + i] * x[t + j] for t in range(n_snap)) / n_snap for j in range(m)] for i in range(m)])
    r0 = U.covariance(x, m, forward_backward=False)
    assert r0 == pytest.approx(ref, abs=1e-12)
    w, v = U.eigendecomposition(U.covariance(x, m))
    ws, _ = linalg.eigh(U.covariance(x, m))
    assert w == pytest.approx(ws[::-1], abs=1e-12) and np.all(np.diff(w) <= 0)
    assert v.T @ v == pytest.approx(np.eye(m), abs=1e-12)


@pytest.mark.parametrize("freqs", [(0.123,), (0.1, 0.1375), (0.07, 0.2, 0.33)])
def test_noise_free_lines_are_recovered_by_all_three_methods(freqs):
    n = 96
    t = np.arange(n)
    x = sum(np.cos(2 * np.pi * f * t + 0.3 * (k + 1)) for k, f in enumerate(freqs))
    w, v = U.eigendecomposition(U.covariance(x, 32))
    p = 2 * len(freqs)
    assert np.all(w[p:] < 1e-12 * w[0]) and np.all(w[:p] > 1e-3)                                      # genau p von null verschiedene Eigenwerte
    grid = np.arange(1, 8 * n // 2) / (8 * n)
    assert U.esprit(v, p) == pytest.approx(sorted(freqs), abs=1e-8)
    assert U.root_music(v, p) == pytest.approx(sorted(freqs), abs=1e-8)
    assert U.music_frequencies(v, p, len(freqs), grid) == pytest.approx(sorted(freqs), abs=2e-5)
    assert U.estimate_count(np.maximum(w, 1e-9 * w[0]), n - 32 + 1, "MDL") == p                  # Rundungsfehler-Eigenwerte auf einen gemeinsamen Boden gesetzt


def test_root_music_equals_the_music_peaks_in_noise():
    rng = np.random.default_rng(8)
    n = 128
    t = np.arange(n)
    x = np.cos(2 * np.pi * 0.2 * t + 1.0) + np.cos(2 * np.pi * 0.23 * t + 2.0) + 0.3 * rng.normal(size=n)
    w, v = U.eigendecomposition(U.covariance(x, 40))
    grid = np.arange(1, 8 * n // 2) / (8 * n)
    assert U.root_music(v, 4) == pytest.approx(U.music_frequencies(v, 4, 2, grid), abs=3e-4)
    assert U.esprit(v, 4) == pytest.approx(U.music_frequencies(v, 4, 2, grid), abs=1.5e-3)


def test_estimate_count_equals_an_independent_loop_over_the_eigenvalues():
    w = np.sort(np.random.default_rng(6).uniform(0.05, 5.0, size=10))[::-1]
    n_snap, m = 40, 10
    crit_mdl, crit_aic = [], []
    for k in range(m - 1):
        tail = w[k:]
        g, a = np.exp(np.log(tail).mean()), tail.mean()
        ll = -n_snap * (m - k) * math.log(g / a)
        crit_mdl.append(ll + 0.5 * k * (2 * m - k) * math.log(n_snap))
        crit_aic.append(2 * ll + 2 * k * (2 * m - k))
    assert U.estimate_count(w, n_snap, "MDL") == int(np.argmin(crit_mdl)) and U.estimate_count(w, n_snap, "AIC") == int(np.argmin(crit_aic))


def test_local_maxima_peak_frequencies_and_the_parabola():
    v = [0.0, 2.0, 1.0, 1.0, 3.0, 3.0, 0.0, 1.0]
    assert U.local_maxima(v) == [1, 4]                                                              # Plateau: der erste Punkt zählt, Rand nie
    f = np.arange(8) / 10.0
    assert U.peak_frequencies(v, f, 1) == pytest.approx([0.4]) and U.peak_frequencies(v, f, 5) == pytest.approx([0.1, 0.4])
    # Parabel durch die Logarithmen: y = exp(-(i - 2.3)^2) hat den Scheitel bei 2.3
    y = np.exp(-(np.arange(6) - 2.3) ** 2)
    assert U.refine_parabola(y, 2, 1.0) == pytest.approx(2.3, abs=1e-12) and U.refine_parabola(y, 2, 0.5) == pytest.approx(1.15, abs=1e-12)
    assert U.refine_parabola(np.ones(5), 2, 1.0) == pytest.approx(2.0)                              # flach: Nenner 0


def test_periodogram_frequencies_find_off_grid_lines():
    n = 64
    t = np.arange(n)
    x = np.cos(2 * np.pi * 0.2 * t + 0.4) + np.cos(2 * np.pi * 0.3125 * t + 1.0)
    f = U.periodogram_frequencies(x, 2)
    assert f == pytest.approx([0.2, 0.3125], abs=2e-3)
    assert len(U.periodogram_frequencies(x, 1)) == 1


def test_crb_bins_matches_the_formula_and_scales():
    n, snr = 64, 10.0
    assert U.crb_bins(n, snr) == pytest.approx(math.sqrt(3.0 / (math.pi ** 2 * 10.0 * n * (n * n - 1.0))) * n, rel=1e-12)
    assert U.crb_bins(n, snr + 20.0) == pytest.approx(U.crb_bins(n, snr) / 10.0, rel=1e-12)
    assert U.crb_bins(2 * n, snr) == pytest.approx(U.crb_bins(n, snr) * 2.0 * math.sqrt(n * (n * n - 1.0) / (2 * n * (4.0 * n * n - 1.0))), rel=1e-12)      # ~ n^{-1/2} in Bins


def test_ops_subspace():
    assert U.ops_subspace(64, 21) == (64 - 21 + 1) * 21 ** 2 + 9 * 21 ** 3


def test_snapshot_bounds_allow_two_and_the_full_length():
    x = np.arange(5.0)
    assert U.snapshots(x, 2).shape == (4, 2) and U.snapshots(x, 5).shape == (1, 5)
    assert U.covariance(x, 5).shape == (5, 5)


def test_estimate_count_can_pick_the_largest_allowed_k_and_music_accepts_order_one():
    w = np.array([10.0, 1.0, 1.0])                                    # m = 3: erlaubt sind k = 0 und k = 1; ab k = 1 sind die übrigen gleich
    assert U.estimate_count(w, 100, "MDL") == 1 and U.estimate_count(w, 100, "AIC") == 1
    v = np.eye(3)
    assert U.music_spectrum(v, 1, np.array([0.1, 0.3])).shape == (2,)


def test_estimate_count_never_returns_the_last_index_even_when_its_criterion_is_lowest():
    w = np.array([10.0, 1.0, 1e-6])                                   # k = 2 hätte (nur ein Eigenwert im Rauschteil) das kleinste Kriterium, ist aber nicht erlaubt
    assert U.estimate_count(w, 100, "MDL") == 1 and U.estimate_count(w, 100, "AIC") == 1


def test_local_maxima_at_the_first_and_last_interior_point():
    assert U.local_maxima([0.0, 1.0, 0.0]) == [1] and U.local_maxima([0.0, 1.0, 0.0, 0.0]) == [1] and U.local_maxima([0.0, 0.0, 1.0, 0.0]) == [2]
    assert U.local_maxima([1.0, 0.0, 1.0]) == [] and U.local_maxima([0.0, 1.0]) == []


@pytest.mark.parametrize("strong,expected", [(7, 20), (8, 8), (24, 24), (25, 20)])
def test_periodogram_frequencies_ignore_the_eight_points_at_each_end(strong, expected):
    n = 64                                                            # factor 1: 33 Punkte, zählen dürfen die Indizes 8 bis 24
    t = np.arange(n)
    x = 3.0 * np.cos(2 * np.pi * (strong + 0.3) * t / n + 0.4) + np.cos(2 * np.pi * (20 + 0.3) * t / n + 1.0)
    f = U.periodogram_frequencies(x, 1, factor=1)
    assert f[0] * n == pytest.approx(expected, abs=1.0)
