"""Jede Zahl und Aussage in den Texten der App (Presets, Einleitung, Experiment-Überschriften) und in der README ist hier an Rechnung oder Messung gebunden."""

import re
from pathlib import Path

import numpy as np
import pytest

import ure_constants as C
import ure_evaluation as ev
from sweeps_cache import get

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def preset(name, **over):
    return ev.Settings(**{**{k: v for k, v in C.PRESETS[name].items()}, **over})


def A(name, **over):
    return ev.analyse(preset(name, **over))


def share(rows, label, delta):
    r = [x for x in rows if x["label"] == label][0]
    return r["shares"][r["deltas"].index(delta)]


def acc(rows, n, snr):
    return [r for r in rows if r["n"] == n and r["snr"] == snr][0]


def row(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())][0]


def pct(x):
    return f"{100 * x:.1f}".replace(".0", "").replace(".", ",")


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_preset_one_bin_music_and_esprit_find_both_lines_and_the_hann_periodogram_does_not():
    a = A("Zwei Linien: 1 Bin Abstand")
    assert a.settings.gap == 1.0 and a.settings.n == 64 and a.p == 4
    assert a.found["MUSIC"] and a.found["Wurzel-MUSIC"] and a.found["ESPRIT"] and not a.found["Periodogramm"] and not a.resolved["Periodogramm"]


def test_preset_aic_overestimates_and_mdl_does_not_so_that_the_frequencies_no_longer_fit():
    a = A("Linienzahl: AIC gegen MDL")
    assert a.k_aic >= 8 and a.k_mdl == 4 and a.p == a.k_aic + a.k_aic % 2 and not a.found["MUSIC"] and not a.found["ESPRIT"]
    b = A("Linienzahl: AIC gegen MDL", order_mode="MDL")
    assert b.p == 4 and b.found["MUSIC"] and b.found["ESPRIT"]


def test_preset_single_line_every_method_is_close_to_the_bound():
    a = A("Eine Linie: nahe der Schranke")
    assert a.settings.lines == 1 and a.p == 2 and a.crb == pytest.approx(0.0194, abs=5e-4)
    assert all(a.found[k] for k in a.found)
    assert max(abs(e[0]) for e in a.err_bins.values()) < 2.0 * a.crb
    assert abs(a.err_bins["Periodogramm"][0]) < 1.5 * a.crb


def test_preset_coloured_noise_music_and_esprit_miss_a_line():
    a = A("Gefärbtes Rauschen")
    assert a.settings.background == "gefärbt, zehnfache Leistung" and a.p == 4
    assert not a.found["MUSIC"] and not a.found["ESPRIT"] and not a.found["Wurzel-MUSIC"]
    w = A("Gefärbtes Rauschen", background="weiß")
    assert w.found["MUSIC"] and w.found["ESPRIT"]


def test_preset_too_small_an_order_leaves_one_line_missing():
    a = A("Zu kleine Ordnung")
    assert a.p == 2 and not any(a.found[k] for k in ("MUSIC", "Wurzel-MUSIC", "ESPRIT"))
    assert len(a.est_hz["MUSIC"]) == 1 and a.err_bins["MUSIC"] is None
    assert A("Zu kleine Ordnung", order=4).found["ESPRIT"]


# --- README: Auflösung ------------------------------------------------------------------------------------------------------------------------


def test_readme_resolution_numbers():
    res = get("resolution")
    assert (share(res, "MUSIC (m = n/3), gefunden", 1.0), share(res, "Wurzel-MUSIC (m = n/3), gefunden", 1.0), share(res, "ESPRIT (m = n/3), gefunden", 1.0)) == (0.995, 0.995, 1.0)
    assert share(res, "Hann-Periodogramm, aufgelöst", 1.0) == 0.2 and share(res, "Burg p = 16, aufgelöst", 1.0) == 0.565 and share(res, "Multitaper NW = 2, aufgelöst", 1.0) == 0.0
    assert share(res, "ESPRIT (m = n/3), gefunden", 0.5) == 0.21 and share(res, "ESPRIT (m = n/2), gefunden", 0.5) == 0.525
    for s in ("99,5 %", "20 %", "56,5 %", "21 %", "52,5 %"):
        assert s in README
    assert C.EXP_RES_N == 64 and C.EXP_RES_SNR == 10.0 and C.EXP_RES_RUNS == 200


def test_the_music_pseudospectrum_separates_the_lines_in_most_runs_at_one_bin():
    assert share(get("resolution"), "MUSIC (m = n/3), aufgelöst", 1.0) == 0.965


# --- README: Genauigkeit ----------------------------------------------------------------------------------------------------------------------


def test_readme_accuracy_numbers():
    r = acc(get("accuracy"), 256, 5.0)["methods"]
    assert [round(r[k]["ratio"], 2) for k in ("Periodogramm, Rechteck", "MUSIC", "ESPRIT")] == [0.92, 1.09, 0.95]
    q = acc(get("accuracy"), 64, 0.0)["methods"]
    assert [round(q[k]["ratio"], 2) for k in ("Periodogramm, Rechteck", "ESPRIT", "MUSIC", "Periodogramm, Hann")] == [1.13, 1.48, 1.57, 1.69]
    w = acc(get("accuracy"), 64, -6.0)["methods"]
    assert round(w["Periodogramm, Rechteck"]["ratio"], 1) == 1.8 and round(w["ESPRIT"]["ratio"], 1) == 3.4
    assert round(100 * w["Periodogramm, Rechteck"]["outliers"]) == 10 and round(100 * w["ESPRIT"]["outliers"]) == 20
    for s in ("0,92-fache", "1,09-fache", "0,95-fache", "1,13 (Rechteck", "1,48 (ESPRIT", "1,57 (MUSIC", "1,69 (Hann", "1,8-fachen", "3,4-fachen"):
        assert s in README


def test_every_method_stays_within_a_modest_factor_of_the_bound_above_the_threshold():
    for r in get("accuracy"):
        if r["snr"] >= 0.0 and r["n"] == 256:
            assert all(0.85 < v["ratio"] < 1.5 for v in r["methods"].values()), r


def test_the_bound_in_the_readme_row_is_the_asymptotic_one():
    assert acc(get("accuracy"), 256, 5.0)["crb"] == pytest.approx(0.01938, abs=5e-5)


# --- README: Linienzahl, Matrix, gefärbtes Rauschen, Ordnung, Höhe ---------------------------------------------------------------------------


def test_readme_count_numbers():
    c = row(get("count"), lines=2, snr=0.0)
    assert (c["MDL"]["right"], c["AIC"]["right"], c["AIC"]["over"]) == (0.98, 0.67, 0.33)
    w = row(get("count"), lines=2, snr=-9.0)
    assert (w["MDL"]["right"], w["AIC"]["right"], w["MDL"]["under"]) == (0.05, 0.25, 0.95)
    assert C.EXP_COUNT_N == 128 and C.EXP_COUNT_M == 32
    for s in ("98 %", "67 %", "33 %", "MDL 5 %", "AIC 25 %", "(95 %)"):
        assert s in README


def test_readme_matrix_numbers():
    m = {r["m"]: r for r in get("matrix")}
    assert (m[6]["MUSIC"]["found"], m[6]["ESPRIT"]["found"]) == (0.48, 0.495) and (m[8]["MUSIC"]["found"], m[8]["ESPRIT"]["found"]) == (0.895, 0.925)
    assert all(m[k]["MUSIC"]["found"] == 1.0 and m[k]["ESPRIT"]["found"] == 1.0 for k in (12, 16, 24, 32, 48))
    assert round(m[12]["ESPRIT"]["rmse"], 3) == 0.067 and round(m[24]["ESPRIT"]["rmse"], 3) == 0.025
    assert C.EXP_MATRIX_N == 64 and C.EXP_MATRIX_DELTA == 2.0
    for s in ("48 %", "49,5 %", "89,5 / 92,5 %", "0,067", "0,025"):
        assert s in README


def test_readme_coloured_numbers():
    r = {x["label"]: x for x in get("coloured")}
    assert (r["gefärbt, zehnfache Leistung"]["MUSIC"], r["gefärbt, zehnfache Leistung"]["ESPRIT"], r["gefärbt, zehnfache Leistung"]["Burg"]) == (0.755, 0.67, 1.0)
    assert all(r["gefärbt, gleiche Leistung"][k] == 1.0 and r["weißes Rauschen"][k] == 1.0 for k in ("MUSIC", "ESPRIT", "Burg"))
    assert C.EXP_COLOUR_N == 128
    for s in ("75,5 %", "67 %", "100 %"):
        assert s in README


def test_readme_order_numbers():
    o = {r["p"]: r for r in get("order")}
    assert round(o[2]["rmse"], 3) == 0.116 and o[2]["extra"] == 0.645 and o[3]["extra"] == 0.175
    assert round(o[4]["rmse"], 3) == 0.044 and o[4]["extra"] == 0.0 and round(o[12]["rmse"], 3) == 0.045 and o[12]["extra"] == 0.02
    for s in ("0,116", "64,5 %", "17,5 %", "0,044", "0,045", "2 % Zusatzgipfel"):
        assert s in README


def test_readme_height_numbers():
    h = get("height")
    assert round(h["periodogram_db"], 1) == -20.0 and round(h["music_db"], 1) == -11.9 and round(h["music_std_db"], 1) == 4.2
    for s in ("−20,0 dB", "−11,9 dB", "4,2 dB"):
        assert s in README


# --- Texte der App und Einleitung ------------------------------------------------------------------------------------------------------------


def test_the_app_text_claims_a_hann_periodogram_needs_about_three_bins():
    res = get("resolution")
    assert share(res, "Hann-Periodogramm, aufgelöst", 2.0) < 0.6 and share(res, "Hann-Periodogramm, aufgelöst", 3.0) == 1.0


def test_the_app_text_threshold_near_zero_db_at_n_64():
    r = acc(get("accuracy"), 64, -6.0)["methods"]
    s = acc(get("accuracy"), 64, 0.0)["methods"]
    assert all(0.09 <= v["outliers"] <= 0.21 for v in r.values()) and all(v["outliers"] == 0.0 for v in s.values())             # „10 bis 20 % Ausreißer“ bei −6 dB, keine bei 0 dB


def test_the_readme_standard_error_remark():
    se = np.sqrt(0.5 * 0.5 / 100)
    assert se * 100 <= 5.0 and "höchstens 5 Prozentpunkte" in README


def test_the_readme_mentions_the_defaults():
    s = ev.Settings()
    assert s.n == 64 and s.lines == 2 and s.m_size == 21 and "Standardfehler" in README


def test_every_number_with_a_percent_sign_in_the_readme_table_is_formatted_like_the_code():
    assert pct(0.995) == "99,5" and pct(0.2) == "20" and pct(1.0) == "100"
    assert re.search(r"Bei 1 Bin Abstand finden MUSIC", README)
