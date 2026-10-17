"""Einstellungen, Signalaufbau und Analyse: Regeln der Linienlagen, Matrixgröße, Maßstab der Linienstärke, Invarianten der Analyse."""

import math

import numpy as np
import pytest

import signal_generator as G
import ure_constants as C
import ure_evaluation as ev


def test_matrix_size_rules():
    assert ev.matrix_size(64, "n/3") == 21 and ev.matrix_size(64, "n/2") == 32 and ev.matrix_size(128, "n/4") == 32 and ev.matrix_size(512, "n/6") == 85
    assert ev.matrix_size(32, "n/6") == 12 and ev.matrix_size(32, "n/3") == 12 and ev.matrix_size(32, "n/2") == 16
    assert ev.Settings().m_size == 21
    for n in C.N_CHOICES:
        for choice in C.M_CHOICES:
            m = ev.matrix_size(n, choice)
            assert 12 <= m <= n // 2


def test_line_positions_by_hand():
    s = ev.Settings(n=64, lines=3, line=0.2, gap=2.0)
    assert s.f1_hz == pytest.approx(200.0) and s.f2_hz == pytest.approx(231.25) and s.f3_hz == pytest.approx(330.0)
    assert s.line_hz == pytest.approx([200.0, 231.25, 330.0])
    assert ev.Settings(lines=1).line_hz == pytest.approx([200.0]) and ev.Settings(lines=2).line_hz == pytest.approx([200.0, 231.25])
    high = ev.Settings(n=64, lines=3, line=0.4, gap=6.0)
    assert high.f2_hz == pytest.approx(400.0 - 93.75) and high.f3_hz == pytest.approx(270.0) and high.line_hz == sorted(high.line_hz)


def test_the_second_and_third_line_never_leave_the_band():
    for n in C.N_CHOICES:
        for line in (C.LINE_MIN, 0.2, 0.37, 0.38, C.LINE_MAX):
            for gap in (C.GAP_MIN, 3.0, C.GAP_MAX):
                s = ev.Settings(n=n, line=line, gap=gap, lines=3)
                for f in s.line_hz:
                    assert 0.0 < f <= C.FS / 2 - 2.0 * C.FS / n + 1e-9


def test_the_upper_edge_is_still_allowed_exactly_two_bins_below_nyquist():
    n = 64
    edge = C.FS / 2 - 2.0 * C.FS / n                                   # 437,5 Hz
    s = ev.Settings(n=n, line=(edge - 31.25) / C.FS, gap=2.0)
    assert s.f2_hz == pytest.approx(edge)
    s = ev.Settings(n=n, line=(edge - 31.25 + 0.5) / C.FS, gap=2.0)
    assert s.f2_hz < s.f1_hz


def test_amplitude_gives_the_snr_per_line_against_unit_noise():
    assert ev.amplitude(10) == pytest.approx(math.sqrt(20.0)) and ev.amplitude(0) == pytest.approx(math.sqrt(2.0)) and ev.amplitude(-10) == pytest.approx(math.sqrt(0.2))


def test_the_signal_has_the_requested_power_for_white_and_coloured_backgrounds():
    def mean_power(background, snr):
        return np.mean([np.mean(ev.build_signal(ev.Settings(n=512, lines=1, snr=snr, background=background, seed=s)).x ** 2) for s in range(30)])
    assert mean_power("weiß", 10) == pytest.approx(1.0 + 10.0, rel=0.1)
    assert mean_power("weiß", -10) == pytest.approx(1.0 + 0.1, rel=0.1)
    assert mean_power("gefärbt, gleiche Leistung", -10) == pytest.approx(0.1 + 1.0, rel=0.3)
    assert mean_power("gefärbt, zehnfache Leistung", -10) == pytest.approx(0.1 + 10.0, rel=0.3)


def test_the_analysis_is_deterministic_and_the_seed_changes_the_data():
    a, b = ev.analyse(ev.Settings()), ev.analyse(ev.Settings())
    assert np.array_equal(a.x, b.x) and np.array_equal(a.music, b.music) and a.est_hz == b.est_hz
    assert not np.array_equal(a.x, ev.analyse(ev.Settings(seed=40)).x)


def test_analysis_shapes_and_sorted_eigenvalues():
    a = ev.analyse(ev.Settings(n=128, m="n/4"))
    assert a.m == 32 and a.n_snap == 97 and len(a.eig) == 32 and np.all(np.diff(a.eig) <= 0) and len(a.x) == 128
    assert len(a.f_hz) == len(a.music) == len(a.fft) == len(a.burg) == C.GRID * 128 // 2 - 1
    assert a.f_hz[0] == pytest.approx(C.FS / (C.GRID * 128)) and a.f_hz[-1] < C.FS / 2
    assert a.burg_order == 16 and ev.analyse(ev.Settings(n=32)).burg_order == 10
    assert a.ops == 97 * 32 ** 2 + 9 * 32 ** 3 and a.ops_fft == 64 * 7


def test_fixed_and_estimated_orders():
    for p in C.ORDER_CHOICES:
        a = ev.analyse(ev.Settings(order=p))
        assert a.p == p and len(a.est_hz["ESPRIT"]) == p // 2
    aic, mdl = ev.analyse(ev.Settings(order_mode="AIC")), ev.analyse(ev.Settings(order_mode="MDL"))
    assert aic.p == max(2, aic.k_aic + aic.k_aic % 2) and mdl.p == max(2, mdl.k_mdl + mdl.k_mdl % 2) and mdl.p == 4
    assert ev.analyse(ev.Settings(order_mode="fest", order=8, n=32)).p == 8 and ev.analyse(ev.Settings(order_mode="fest", order=8, n=32)).m == 12


def test_the_order_never_exceeds_the_matrix_minus_two():
    a = ev.analyse(ev.Settings(n=32, background="gefärbt, zehnfache Leistung", order_mode="AIC", snr=-10))
    assert a.p <= a.m - 2


def test_errors_are_in_bins_and_signed():
    a = ev.analyse(ev.Settings(n=64, lines=1, snr=30, order=2))
    e = a.err_bins["ESPRIT"][0]
    assert e == pytest.approx((a.est_hz["ESPRIT"][0] - 200.0) * 64 / C.FS) and abs(e) < 0.01


def test_found_needs_the_right_count_and_a_quarter_of_the_spacing():
    s = ev.Settings(n=128, lines=2, gap=4.0, snr=30)
    a = ev.analyse(s)
    assert all(a.found[k] for k in ("MUSIC", "Wurzel-MUSIC", "ESPRIT"))
    wrong = ev.analyse(ev.Settings(n=128, lines=2, gap=4.0, snr=30, order=2))
    assert not wrong.found["ESPRIT"] and wrong.err_bins["ESPRIT"] is None
    one = ev.analyse(ev.Settings(n=64, lines=1, snr=30, order=2))
    assert one.found["MUSIC"]                                                              # eine Linie: Toleranz ein halbes Bin


def test_resolved_and_crb_exist_only_in_the_matching_cases():
    assert ev.analyse(ev.Settings(lines=2)).resolved is not None and ev.analyse(ev.Settings(lines=2)).crb is None
    one = ev.analyse(ev.Settings(lines=1))
    assert one.resolved is None and one.crb == pytest.approx(ev.U.crb_bins(64, 10.0))
    three = ev.analyse(ev.Settings(lines=3))
    assert three.resolved is None and three.crb is None


def test_generator_stream_is_the_documented_one():
    s = ev.Settings(n=64, lines=2, snr=10, seed=7)
    rng = G.SplitMix64(7)
    ph = 2.0 * math.pi * rng.uniform(3)
    ref = G.make_signal(64, C.FS, [(200.0, math.sqrt(20.0), float(ph[0])), (231.25, math.sqrt(20.0), float(ph[1]))], (), 0.0, 1.0, rng)
    assert np.array_equal(ev.build_signal(s).x, ref.x)


def test_sweeps_accept_small_parameters():
    r = ev.sweep_resolution(n=64, deltas=(1.0, 3.0), runs=4)
    assert len(r) == 9 and all(len(x["shares"]) == 2 for x in r)
    assert len(ev.sweep_accuracy(ns=(32,), snrs=(10.0,), runs=3)) == 1
    assert len(ev.sweep_count(n=64, m=16, lines=((1, (0.2,)),), snrs=(10.0,), runs=3)) == 1
    assert len(ev.sweep_matrix(n=64, ms=(8, 12), runs=3)) == 2
    assert len(ev.sweep_coloured(n=64, m=20, runs=3)) == 3
    assert len(ev.sweep_order(n=64, m=20, ps=(2, 4), runs=3)) == 2
    assert ev.sweep_height(n=64, m=20, runs=3)["runs"] == 3
