"""Randfälle: kleinste und größte Länge, alle Matrixgrößen, Ordnungsarten und Hintergründe, Linien an den Reglergrenzen, Abbildungen und PDF."""

import numpy as np
import pytest

import ure_constants as C
import ure_evaluation as ev
import ure_visualization as V
from ure_pdf_export import build_pdf


@pytest.mark.parametrize("background", list(C.BACKGROUNDS))
@pytest.mark.parametrize("m", list(C.M_CHOICES))
def test_every_background_and_matrix_size_runs(background, m):
    a = ev.analyse(ev.Settings(n=64, background=background, m=m))
    assert np.isfinite(a.music).all() and (a.music > 0).all() and np.isfinite(a.eig).all() and all(np.isfinite(v).all() for v in a.est_hz.values())


@pytest.mark.parametrize("mode", C.ORDER_MODES)
def test_every_order_mode_runs_at_the_shortest_and_the_longest_record(mode):
    for n in (32, 512):
        for lines in C.LINE_COUNTS:
            a = ev.analyse(ev.Settings(n=n, lines=lines, order_mode=mode, order=4))
            assert np.isfinite(a.music).all() and 2 <= a.p <= a.m - 2


@pytest.mark.parametrize("line", [C.LINE_MIN, C.LINE_MAX])
@pytest.mark.parametrize("gap", [C.GAP_MIN, C.GAP_MAX])
def test_lines_at_the_slider_limits_stay_below_nyquist(line, gap):
    for n in (32, 512):
        for lines in C.LINE_COUNTS:
            s = ev.Settings(n=n, line=line, gap=gap, lines=lines)
            assert all(0 < f < C.FS / 2 for f in s.line_hz)
            assert abs(s.f2_hz - s.f1_hz) == pytest.approx(gap * C.FS / n)
            assert np.isfinite(ev.analyse(s).music).all()


def test_extreme_levels_and_seed_limits():
    for snr in (C.SNR_MIN, C.SNR_MAX):
        for lines in C.LINE_COUNTS:
            a = ev.analyse(ev.Settings(lines=lines, snr=snr))
            assert np.isfinite(a.music).all() and all(np.isfinite(v).all() for v in a.est_hz.values())
    for seed in (0, C.SEED_MAX):
        assert np.isfinite(ev.analyse(ev.Settings(seed=seed)).music).all()


def test_pure_noise_level_still_produces_a_valid_count():
    a = ev.analyse(ev.Settings(snr=C.SNR_MIN, order_mode="MDL"))
    assert 2 <= a.p <= a.m - 2 and a.k_mdl >= 0


def test_two_lines_that_nearly_coincide_do_not_crash():
    a = ev.analyse(ev.Settings(n=32, lines=3, gap=4.0))
    assert np.isfinite(a.music).all() and len(a.true_hz) == 3


def test_pdf_builds_for_all_line_counts():
    for lines in C.LINE_COUNTS:
        pdf = build_pdf(ev.analyse(ev.Settings(lines=lines)))
        assert pdf[:4] == b"%PDF" and len(pdf) > 1000
    assert build_pdf(ev.analyse(ev.Settings(lines=2, order=2)))[:4] == b"%PDF"


def test_every_figure_builds():
    a = ev.analyse(ev.Settings())
    one = ev.analyse(ev.Settings(lines=1, order=2))
    for fig in (V.build_spectrum(a), V.build_spectrum(a, zoom=True), V.build_spectrum(one, zoom=True), V.build_eigenvalues(a), V.build_errors(a), V.build_errors(one)):
        assert len(fig.data) >= 1
    empty = V.build_errors(ev.analyse(ev.Settings(order=2, lines=3)))
    assert len(empty.data) == 0 and "stimmt bei keinem Verfahren" in empty.layout.annotations[0].text
    assert len(V.build_errors(one).layout.annotations) == 0 and len(V.build_errors(one).layout.shapes) == 2 and len(V.build_errors(a).layout.shapes) == 0
    assert V.build_height({"true_db": -20.0, "periodogram_db": -20.0, "music_db": -12.0, "music_std_db": 4.0}).data[0].x == ("wahr", "Periodogramm (Hann)", "MUSIC-Pseudospektrum")
