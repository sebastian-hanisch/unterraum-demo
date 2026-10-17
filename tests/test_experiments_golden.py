"""Festwerte der Experimente (feste Zufallsströme): Das Ergebnis ist bei jedem Lauf dasselbe; Abweichungen bedeuten eine geänderte Rechnung."""

import pytest

from sweeps_cache import get

RESOLUTION = {'MUSIC (m = n/3), gefunden': [0.105, 0.755, 0.995, 1.0, 1.0, 1.0, 1.0], 'Wurzel-MUSIC (m = n/3), gefunden': [0.225, 0.8, 0.995, 1.0, 1.0, 1.0, 1.0], 'ESPRIT (m = n/3), gefunden': [0.21, 0.855, 1.0, 1.0, 1.0, 1.0, 1.0], 'MUSIC (m = n/2), gefunden': [0.35, 0.915, 0.995, 1.0, 1.0, 1.0, 1.0], 'ESPRIT (m = n/2), gefunden': [0.525, 0.935, 0.995, 1.0, 1.0, 1.0, 1.0], 'MUSIC (m = n/3), aufgelöst': [0.05, 0.55, 0.965, 1.0, 1.0, 1.0, 1.0], 'Burg p = 16, aufgelöst': [0.005, 0.165, 0.565, 1.0, 1.0, 1.0, 1.0], 'Multitaper NW = 2, aufgelöst': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.995], 'Hann-Periodogramm, aufgelöst': [0.09, 0.085, 0.2, 0.28, 0.495, 1.0, 1.0]}

ACCURACY = {(64, -6.0): {'Periodogramm, Rechteck': (1.798256, 0.1), 'Periodogramm, Hann': (4.698698, 0.17), 'MUSIC': (3.529809, 0.12), 'Wurzel-MUSIC': (3.539296, 0.14), 'ESPRIT': (3.403087, 0.2)}, (64, -3.0): {'Periodogramm, Rechteck': (1.181634, 0.0), 'Periodogramm, Hann': (4.011045, 0.02), 'MUSIC': (1.997584, 0.0), 'Wurzel-MUSIC': (1.996953, 0.0), 'ESPRIT': (2.131166, 0.0)}, (64, 0.0): {'Periodogramm, Rechteck': (1.132557, 0.0), 'Periodogramm, Hann': (1.686411, 0.0), 'MUSIC': (1.569363, 0.0), 'Wurzel-MUSIC': (1.569201, 0.0), 'ESPRIT': (1.481966, 0.0)}, (64, 5.0): {'Periodogramm, Rechteck': (1.119855, 0.0), 'Periodogramm, Hann': (1.652109, 0.0), 'MUSIC': (1.354681, 0.0), 'Wurzel-MUSIC': (1.354531, 0.0), 'ESPRIT': (1.270138, 0.0)}, (64, 10.0): {'Periodogramm, Rechteck': (1.144069, 0.0), 'Periodogramm, Hann': (1.635574, 0.0), 'MUSIC': (1.289609, 0.0), 'Wurzel-MUSIC': (1.289861, 0.0), 'ESPRIT': (1.206903, 0.0)}, (64, 20.0): {'Periodogramm, Rechteck': (1.546414, 0.0), 'Periodogramm, Hann': (1.621882, 0.0), 'MUSIC': (1.253214, 0.0), 'Wurzel-MUSIC': (1.252148, 0.0), 'ESPRIT': (1.168705, 0.0)}, (256, -6.0): {'Periodogramm, Rechteck': (0.988673, 0.0), 'Periodogramm, Hann': (1.45622, 0.0), 'MUSIC': (1.326519, 0.0), 'Wurzel-MUSIC': (1.326585, 0.0), 'ESPRIT': (1.222673, 0.0)}, (256, -3.0): {'Periodogramm, Rechteck': (0.954269, 0.0), 'Periodogramm, Hann': (1.427404, 0.0), 'MUSIC': (1.199056, 0.0), 'Wurzel-MUSIC': (1.198985, 0.0), 'ESPRIT': (1.042726, 0.0)}, (256, 0.0): {'Periodogramm, Rechteck': (0.936376, 0.0), 'Periodogramm, Hann': (1.414308, 0.0), 'MUSIC': (1.137712, 0.0), 'Wurzel-MUSIC': (1.137673, 0.0), 'ESPRIT': (0.985365, 0.0)}, (256, 5.0): {'Periodogramm, Rechteck': (0.919269, 0.0), 'Periodogramm, Hann': (1.40522, 0.0), 'MUSIC': (1.093316, 0.0), 'Wurzel-MUSIC': (1.093172, 0.0), 'ESPRIT': (0.953641, 0.0)}, (256, 10.0): {'Periodogramm, Rechteck': (0.910673, 0.0), 'Periodogramm, Hann': (1.402093, 0.0), 'MUSIC': (1.076353, 0.0), 'Wurzel-MUSIC': (1.076379, 0.0), 'ESPRIT': (0.945617, 0.0)}, (256, 20.0): {'Periodogramm, Rechteck': (0.983245, 0.0), 'Periodogramm, Hann': (1.400367, 0.0), 'MUSIC': (1.070949, 0.0), 'Wurzel-MUSIC': (1.066287, 0.0), 'ESPRIT': (0.9436, 0.0)}}

COUNT = {(1, -9.0): {'AIC': (0.44, 0.36, 0.2), 'MDL': (0.14, 0.86, 0.0)}, (1, -6.0): {'AIC': (0.7, 0.04, 0.26), 'MDL': (0.665, 0.315, 0.02)}, (1, -3.0): {'AIC': (0.73, 0.0, 0.27), 'MDL': (0.97, 0.01, 0.02)}, (1, 0.0): {'AIC': (0.73, 0.0, 0.27), 'MDL': (0.98, 0.0, 0.02)}, (1, 5.0): {'AIC': (0.73, 0.0, 0.27), 'MDL': (0.98, 0.0, 0.02)}, (1, 10.0): {'AIC': (0.73, 0.0, 0.27), 'MDL': (0.98, 0.0, 0.02)}, (2, -9.0): {'AIC': (0.25, 0.565, 0.185), 'MDL': (0.05, 0.95, 0.0)}, (2, -6.0): {'AIC': (0.615, 0.065, 0.32), 'MDL': (0.495, 0.5, 0.005)}, (2, -3.0): {'AIC': (0.67, 0.0, 0.33), 'MDL': (0.97, 0.01, 0.02)}, (2, 0.0): {'AIC': (0.67, 0.0, 0.33), 'MDL': (0.98, 0.0, 0.02)}, (2, 5.0): {'AIC': (0.67, 0.0, 0.33), 'MDL': (0.98, 0.0, 0.02)}, (2, 10.0): {'AIC': (0.67, 0.0, 0.33), 'MDL': (0.98, 0.0, 0.02)}, (3, -9.0): {'AIC': (0.255, 0.605, 0.14), 'MDL': (0.02, 0.98, 0.0)}, (3, -6.0): {'AIC': (0.605, 0.06, 0.335), 'MDL': (0.365, 0.625, 0.01)}, (3, -3.0): {'AIC': (0.62, 0.0, 0.38), 'MDL': (0.94, 0.02, 0.04)}, (3, 0.0): {'AIC': (0.625, 0.0, 0.375), 'MDL': (0.96, 0.0, 0.04)}, (3, 5.0): {'AIC': (0.625, 0.0, 0.375), 'MDL': (0.96, 0.0, 0.04)}, (3, 10.0): {'AIC': (0.625, 0.0, 0.375), 'MDL': (0.96, 0.0, 0.04)}}

MATRIX = {6: (0.48, 0.2526, 0.495, 0.253355), 8: (0.895, 0.194562, 0.925, 0.187455), 12: (1.0, 0.072633, 1.0, 0.067364), 16: (1.0, 0.043976, 1.0, 0.039285), 24: (1.0, 0.029449, 1.0, 0.025199), 32: (1.0, 0.028318, 1.0, 0.025066), 48: (1.0, 0.029679, 1.0, 0.029322)}

COLOURED = {'weißes Rauschen': (1.0, 1.0, 1.0), 'gefärbt, gleiche Leistung': (1.0, 1.0, 1.0), 'gefärbt, zehnfache Leistung': (0.755, 0.67, 1.0)}

ORDER = {2: (0.11574, 0.055, 0.645), 3: (0.047451, 0.0, 0.175), 4: (0.044088, 0.0, 0.0), 5: (0.044141, 0.0, 0.0), 6: (0.044194, 0.0, 0.005), 8: (0.044913, 0.0, 0.02), 12: (0.044581, 0.0, 0.02)}

HEIGHT = {'runs': 100, 'true_db': -20.0, 'periodogram_db': -20.00018, 'music_db': -11.931756, 'music_std_db': 4.157835}



def test_resolution_golden():
    rows = {r["label"]: r for r in get("resolution")}
    assert set(rows) == set(RESOLUTION)
    for label, shares in RESOLUTION.items():
        assert rows[label]["deltas"] == [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        assert rows[label]["shares"] == pytest.approx(shares, abs=1e-6), label


def test_accuracy_golden():
    rows = get("accuracy")
    assert len(rows) == len(ACCURACY) == 12
    for r in rows:
        for k, (ratio, out) in ACCURACY[(r["n"], r["snr"])].items():
            assert r["methods"][k]["ratio"] == pytest.approx(ratio, abs=1e-5) and r["methods"][k]["outliers"] == pytest.approx(out, abs=1e-6), (r["n"], r["snr"], k)
        assert r["runs"] == 100


def test_count_golden():
    rows = get("count")
    assert len(rows) == len(COUNT) == 18
    for r in rows:
        for k, (right, under, over) in COUNT[(r["lines"], r["snr"])].items():
            assert (r[k]["right"], r[k]["under"], r[k]["over"]) == pytest.approx((right, under, over), abs=1e-6), (r["lines"], r["snr"], k)
            assert r[k]["right"] + r[k]["under"] + r[k]["over"] == pytest.approx(1.0, abs=1e-9)


def test_matrix_golden():
    rows = get("matrix")
    assert [r["m"] for r in rows] == list(MATRIX)
    for r in rows:
        mf, mr, ef, er = MATRIX[r["m"]]
        assert (r["MUSIC"]["found"], r["MUSIC"]["rmse"], r["ESPRIT"]["found"], r["ESPRIT"]["rmse"]) == pytest.approx((mf, mr, ef, er), abs=1e-5)


def test_coloured_golden():
    rows = {r["label"]: r for r in get("coloured")}
    assert list(rows) == list(COLOURED)
    for label, (music, esprit, burg) in COLOURED.items():
        assert (rows[label]["MUSIC"], rows[label]["ESPRIT"], rows[label]["Burg"]) == pytest.approx((music, esprit, burg), abs=1e-9)


def test_order_golden():
    rows = get("order")
    assert [r["p"] for r in rows] == list(ORDER)
    for r in rows:
        assert (r["rmse"], r["outliers"], r["extra"]) == pytest.approx(ORDER[r["p"]], abs=1e-5)


def test_height_golden():
    h = get("height")
    for k, v in HEIGHT.items():
        assert h[k] == pytest.approx(v, abs=1e-4), k
