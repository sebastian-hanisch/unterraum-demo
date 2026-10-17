"""Vergleichsverfahren (wortgleich aus den Vorgänger-Demos): Fenster, Slepian-Taper, Multitaper, Burg, AR-Spektrum, Auflösungskriterium."""

import numpy as np
import pytest
from scipy.signal import windows as spw

import signal_generator as G
import ure_compare as H


def test_windows_and_multitaper_by_a_direct_loop():
    assert H.window("Rechteck", 4).tolist() == [1.0] * 4 and H.window("Hann", 4) == pytest.approx([0.0, 0.5, 1.0, 0.5])
    with pytest.raises(ValueError):
        H.window("Gauß", 4)
    h = H.dpss(16, 2, 3)
    x = np.random.default_rng(4).normal(size=16)
    y = H.eigenspectra(x, h, 32)
    est = H.multitaper_equal(x, h, 2.0, 32)
    for k in range(3):
        for j in (0, 5, 16):
            assert y[k, j] == pytest.approx(sum(h[k, t] * x[t] * np.exp(-2j * np.pi * j * t / 32) for t in range(16)), abs=1e-12)
    assert est == pytest.approx((np.abs(y) ** 2).mean(axis=0) / 2.0, abs=1e-12)


@pytest.mark.parametrize("n,nw,k", [(3, 1, 3), (16, 2, 3), (64, 2.5, 4), (128, 4, 7), (257, 3, 5), (512, 6, 11)])
def test_dpss_equals_scipy(n, nw, k):
    assert H.dpss(n, nw, k) == pytest.approx(spw.dpss(n, nw, Kmax=k, sym=True, norm=2), abs=1e-9)


def test_dpss_signs_orthonormality_and_validation():
    h = H.dpss(64, 2, 4)
    assert h @ h.T == pytest.approx(np.eye(4), abs=1e-10)
    for k in range(4):
        assert h[k][::-1] == pytest.approx(h[k] if k % 2 == 0 else -h[k], abs=1e-9)
    assert h[0].sum() > 0 and h[2].sum() > 0
    for k in (1, 3):
        assert h[k][h[k] ** 2 > max(1e-7, 1.0 / 64)][0] > 0
    for bad in ((2, 0.4, 1), (1, 0.2, 1), (8, 2, 0), (8, 2, 9), (8, 0, 1), (8, 4, 1), (8, -1, 1), (8, 2, -1)):
        with pytest.raises(ValueError):
            H.dpss(*bad)
    assert H.dpss(3, 1, 3).shape == (3, 3) and H.dpss(8, 3.9, 8).shape == (8, 8) and H.dpss(4, 1.9, 1).shape == (1, 4)


def test_dpss_cache_returns_copies_and_is_cleared_after_sixteen_entries():
    a = H.dpss(32, 2, 2)
    a[0, 0] = 99.0
    assert H.dpss(32, 2, 2)[0, 0] != 99.0
    H._CACHE.clear()
    for n in range(8, 8 + 17):
        H.dpss(n, 1.0, 1)
    assert len(H._CACHE) == 17
    H.dpss(40, 1.0, 1)
    assert len(H._CACHE) == 1


def test_sturm_count_by_hand_and_inverse_iteration_finds_the_eigenvector():
    d = np.array([1.0, 2.0, 3.0])
    e2 = np.array([0.0, 0.0])                                            # Diagonalmatrix: Eigenwerte 1, 2, 3
    assert H._sturm_count(d, e2, np.array([0.5, 1.5, 2.5, 3.5])).tolist() == [0, 1, 2, 3]
    assert H._sturm_count(d, e2, np.array([1.0 + 1e-9, 3.0 - 1e-9])).tolist() == [1, 2]
    dd, ee = np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0])
    v = H._inverse_iteration(dd, ee, 2.0 + np.sqrt(2.0) + 1e-9, np.array([1.0, 0.3, 0.7]))
    assert abs(v[0]) == pytest.approx(0.5, abs=1e-6) and abs(v[1]) == pytest.approx(np.sqrt(0.5), abs=1e-6)


def test_lines_resolved_needs_two_distinct_maxima_and_a_dip():
    spec = np.array([0.0, 4.0, 1.0, 4.0, 0.0])
    assert H.lines_resolved(spec, 1, 1.0, 3.0)
    assert not H.lines_resolved(np.array([0.0, 4.0, 2.0, 4.0, 0.0]), 1, 1.0, 3.0)        # Einbruch gleich depth * Maximum zählt nicht
    assert not H.lines_resolved(np.array([0.0, 4.0, 3.0, 2.0, 0.0]), 1, 1.0, 3.0)
    assert not H.lines_resolved(np.array([0.0, 1.0, 9.0, 1.0, 0.0]), 1, 1.0, 3.0)        # ein Gipfel
    assert H.lines_resolved(np.array([0.0, 4.0, 1.0, 4.0, 0.0]), 1, 3.0, 1.0)            # Reihenfolge der Linien egal


def test_burg_of_the_ar_mini_instance_by_hand():
    x = np.array([1.0, 2.0, 3.0, 4.0])                                                  # Ordnung 1: a = 2 (2 + 6 + 12) / ((4 + 9 + 16) + (1 + 4 + 9)) = 40/43
    a, err, refl = H.burg(x, 1)
    assert a == pytest.approx([40.0 / 43.0], abs=1e-12) and refl == pytest.approx([40.0 / 43.0], abs=1e-12)
    assert err[0] == pytest.approx(7.5) and err[1] == pytest.approx(7.5 * (1 - (40.0 / 43.0) ** 2), abs=1e-12)


def test_burg_keeps_the_reflection_coefficients_inside_the_unit_interval_and_handles_zeros():
    x = np.random.default_rng(1).normal(size=64)
    a, err, refl = H.burg(x, 12)
    assert np.all(np.abs(refl) < 1.0) and np.all(np.diff(err) < 0) and len(a) == 12
    a0, err0, refl0 = H.burg(np.zeros(20), 3)
    assert np.all(a0 == 0.0) and np.all(np.isfinite(err0)) and np.all(refl0 == 0.0)


def test_burg_equals_an_independent_stagewise_minimisation():
    x = np.random.default_rng(2).normal(size=40) + np.sin(0.7 * np.arange(40))
    a, err, refl = H.burg(x, 3)
    ef, eb = x.copy(), x.copy()
    ref = []
    for _ in range(3):
        f, b = ef[1:], eb[:-1]
        grid = np.linspace(-0.999, 0.999, 19981)
        k = float(grid[int(np.argmin([((f - kk * b) ** 2).sum() + ((b - kk * f) ** 2).sum() for kk in grid]))])
        ref.append(k)
        ef, eb = f - k * b, b - k * f
    assert refl == pytest.approx(ref, abs=2e-4)


def test_ar_spectrum_of_white_noise_an_ar1_and_the_generators_density():
    f = np.array([0.0, 100.0, 500.0])
    assert H.ar_spectrum([], 2.0, f, 1000.0) == pytest.approx([0.002, 0.002, 0.002], abs=1e-15)
    s = H.ar_spectrum([0.5], 1.0, f, 1000.0)
    assert s[0] == pytest.approx(1.0 / 1000.0 / 0.25, rel=1e-12) and s[2] == pytest.approx(1.0 / 1000.0 / 2.25, rel=1e-12)
    poles = ((0.2, 0.97),)
    ar = G.ar_from_poles(poles)
    sig = G.ar_unit_sigma(ar)
    g = np.linspace(0.0, 500.0, 101)
    assert H.ar_spectrum(ar, sig ** 2, g, 1000.0) == pytest.approx(G.true_psd(g, 1000.0, poles, sig, 0.0), rel=1e-10)
