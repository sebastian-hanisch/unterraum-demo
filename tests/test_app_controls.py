"""AppTest: Auswahlfelder, Permalink, bedingte Regler, PDF (die Rauchtests in test_app.py decken nur Knöpfe und Slider ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import ure_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
GAP = "Abstand der Linie 2 (Bins von f_s/n)"


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _labels(at):
    return [x.label for x in at.slider]


def _order_sliders(at):
    return [x for x in at.select_slider if x.label.startswith("Komponenten p")]


@pytest.mark.parametrize("background", list(C.BACKGROUNDS))
def test_every_background_runs(background):
    at = _run()
    at.selectbox(key="background_select").set_value(background).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("m", list(C.M_CHOICES))
def test_every_matrix_size_runs(m):
    at = _run()
    at.selectbox(key="m_select").set_value(m).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("lines", list(C.LINE_COUNTS))
def test_every_line_count_runs_and_the_gap_slider_needs_two_lines(lines):
    at = _run()
    at.radio(key="lines_select").set_value(lines).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert (GAP in _labels(at)) == (lines >= 2)


def test_the_fixed_order_select_appears_only_for_a_fixed_order():
    at = _run({"om": "MDL"})
    assert len(_order_sliders(at)) == 0
    at.selectbox(key="order_mode_select").set_value("fest").run(timeout=TIMEOUT)
    assert not at.exception and len(_order_sliders(at)) == 1
    at.selectbox(key="order_mode_select").set_value("AIC").run(timeout=TIMEOUT)
    assert not at.exception and len(_order_sliders(at)) == 0


def test_the_fixed_order_and_the_gap_survive_switching_away_and_back():
    at = _run({"om": "fest", "p": "6", "l": "2", "d": "4.5"})
    assert _order_sliders(at)[0].value == 6
    at.selectbox(key="order_mode_select").set_value("MDL").run(timeout=TIMEOUT)
    at.selectbox(key="order_mode_select").set_value("fest").run(timeout=TIMEOUT)
    assert _order_sliders(at)[0].value == 6
    at.radio(key="lines_select").set_value(1).run(timeout=TIMEOUT)
    at.radio(key="lines_select").set_value(2).run(timeout=TIMEOUT)
    assert {x.label: x for x in at.slider}[GAP].value == 4.5


def test_permalink_is_applied_and_invalid_values_are_ignored():
    at = _run({"n": "256", "l": "3", "f": "0.3", "d": "99", "snr": "-99", "bg": "gefärbt, gleiche Leistung", "m": "n/2", "om": "AIC", "p": "6", "seed": "42"})
    assert at.select_slider(key="n_select").value == 256 and at.radio(key="lines_select").value == 3 and at.selectbox(key="background_select").value == "gefärbt, gleiche Leistung"
    assert at.selectbox(key="m_select").value == "n/2" and at.selectbox(key="order_mode_select").value == "AIC" and at.number_input(key="seed_input").value == 42
    by = {x.label: x for x in at.slider}
    assert by["Linie 1 (Bruchteil der Abtastrate)"].value == 0.3 and by[GAP].value == C.GAP_MAX and by["Jede Linie gegen Rauschen (dB je Abtastwert)"].value == C.SNR_MIN
    at2 = _run({"n": "3000", "l": "7", "bg": "unsinn", "m": "n/7", "om": "unsinn", "p": "5"})
    assert at2.select_slider(key="n_select").value == C.DEFAULT_N and at2.radio(key="lines_select").value == C.DEFAULT_LINE_COUNT
    assert at2.selectbox(key="background_select").value == C.DEFAULT_BACKGROUND and at2.selectbox(key="m_select").value == C.DEFAULT_M
    assert at2.selectbox(key="order_mode_select").value == C.DEFAULT_ORDER_MODE


def test_preset_button_sets_the_widgets():
    at = _run()
    at.button(key="preset_Eine Linie: nahe der Schranke").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.radio(key="lines_select").value == 1 and at.select_slider(key="n_select").value == 256 and GAP not in _labels(at)
    at.button(key="preset_Linienzahl: AIC gegen MDL").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="order_mode_select").value == "AIC" and at.number_input(key="seed_input").value == C.PRESETS["Linienzahl: AIC gegen MDL"]["seed"]


def test_pdf_button_builds_a_download():
    at = _run()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any(d.proto.label.startswith("⬇️") for d in at.get("download_button"))


def test_the_dice_button_changes_the_seed():
    at = _run()
    before = at.number_input(key="seed_input").value
    [b for b in at.button if "würfeln" in b.label][0].click().run(timeout=TIMEOUT)
    assert not at.exception and at.number_input(key="seed_input").value != before
