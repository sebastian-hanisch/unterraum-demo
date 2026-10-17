"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle gezeichneten Regler sind in jeder Einstellung wirksam: Der Abstand der Linie 2 erscheint nur bei mindestens zwei Linien, die feste Ordnung nur bei „fest“. Ihre Werte stehen nicht
im Zustand eines Reglers, sondern unter `*_kept` (kein Frontend/Backend-Auseinanderlaufen: der Regler bekommt sie als Startwert).
"""

import random
from dataclasses import dataclass

import streamlit as st

import ure_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_select": SettingSpec("n", int, C.DEFAULT_N, choices=C.N_CHOICES),
    "lines_select": SettingSpec("l", int, C.DEFAULT_LINE_COUNT, choices=C.LINE_COUNTS),
    "line_slider": SettingSpec("f", float, C.DEFAULT_LINE, C.LINE_MIN, C.LINE_MAX),
    "gap_kept": SettingSpec("d", float, C.DEFAULT_GAP, C.GAP_MIN, C.GAP_MAX),
    "snr_slider": SettingSpec("snr", int, C.DEFAULT_SNR, C.SNR_MIN, C.SNR_MAX),
    "background_select": SettingSpec("bg", str, C.DEFAULT_BACKGROUND, choices=C.BACKGROUNDS),
    "m_select": SettingSpec("m", str, C.DEFAULT_M, choices=C.M_CHOICES),
    "order_mode_select": SettingSpec("om", str, C.DEFAULT_ORDER_MODE, choices=C.ORDER_MODES),
    "order_kept": SettingSpec("p", int, C.DEFAULT_ORDER, choices=C.ORDER_CHOICES),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_select", "lines": "lines_select", "line": "line_slider", "gap": "gap_kept", "snr": "snr_slider", "background": "background_select", "m": "m_select",
               "order_mode": "order_mode_select", "order": "order_kept", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
