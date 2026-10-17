"""Presets: Vollständigkeit, gültige Werte, Standardwerte."""

import ure_constants as C
import ure_evaluation as ev
import ure_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_choices_and_bounds():
    for name, p in C.PRESETS.items():
        assert p["n"] in C.N_CHOICES and p["lines"] in C.LINE_COUNTS and p["m"] in C.M_CHOICES and p["order_mode"] in C.ORDER_MODES and p["order"] in C.ORDER_CHOICES, name
        assert p["background"] in C.BACKGROUNDS
        assert C.LINE_MIN <= p["line"] <= C.LINE_MAX and C.GAP_MIN <= p["gap"] <= C.GAP_MAX and C.SNR_MIN <= p["snr"] <= C.SNR_MAX and 0 <= p["seed"] <= C.SEED_MAX


def test_default_settings_are_valid_choices():
    s = ev.Settings()
    assert s.n in C.N_CHOICES and s.lines in C.LINE_COUNTS and s.m in C.M_CHOICES and s.order_mode in C.ORDER_MODES and s.order in C.ORDER_CHOICES and s.background in C.BACKGROUNDS
    assert C.LINE_MIN <= s.line <= C.LINE_MAX and C.GAP_MIN <= s.gap <= C.GAP_MAX and C.SNR_MIN <= s.snr <= C.SNR_MAX


def test_setting_specs_cover_every_preset_key_and_the_defaults_match():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    s = ev.Settings()
    defaults = {"n": s.n, "lines": s.lines, "line": s.line, "gap": s.gap, "snr": s.snr, "background": s.background, "m": s.m, "order_mode": s.order_mode, "order": s.order, "seed": s.seed}
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == defaults[key], key


def test_preset_names_are_short_enough_for_the_buttons_and_url_params_are_unique():
    assert max(len(n) for n in C.PRESETS) <= 32
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_background_factors_cover_the_choices():
    assert set(C.BACKGROUND_FACTOR) == set(C.BACKGROUNDS) and C.BACKGROUND_FACTOR["weiß"] is None
    assert C.BACKGROUND_FACTOR["gefärbt, gleiche Leistung"] == 1.0 and C.BACKGROUND_FACTOR["gefärbt, zehnfache Leistung"] == 10.0
    assert set(C.M_DIVISOR) == set(C.M_CHOICES)
