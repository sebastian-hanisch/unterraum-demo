"""Verdikte: reine Funktionen an handgesetzten Zeilen; Art, Schwellen und jede Zahl im Text."""

import numpy as np

import ure_evaluation as ev


def make_analysis(**kw):
    z = np.zeros(1)
    found = {k: True for k in ("MUSIC", "Wurzel-MUSIC", "ESPRIT", "Periodogramm")}
    base = dict(settings=ev.Settings(), m=21, n_snap=44, p=4, k_aic=4, k_mdl=4, eig=z, f_hz=z, x=z, music=z, fft=z, burg=z, burg_order=16, true_hz=[200.0, 231.25],
                est_hz={k: [200.0, 231.0] for k in found}, err_bins={k: [0.0, 0.0] for k in found}, found=found, crb=None, resolved={"MUSIC": True, "Periodogramm": False, "AR": False},
                ops=1000, ops_fft=100)
    base.update(kw)
    return ev.Analysis(**base)


def test_main_verdict_two_lines_all_found_and_the_periodogram_failing():
    a = make_analysis(found={"MUSIC": True, "Wurzel-MUSIC": True, "ESPRIT": True, "Periodogramm": False})
    kind, text = ev.verdict_main(a)
    assert kind == "success"
    assert text == "4 Komponenten (2 Linien im Modell), Matrix 21 × 21; Linienzahl nach AIC 2, nach MDL 2. MUSIC und ESPRIT finden beide Linien; das Periodogramm trennt sie nicht."


def test_main_verdict_two_lines_all_found_including_the_periodogram_and_a_miss():
    kind, text = ev.verdict_main(make_analysis())
    assert kind == "success" and text.endswith("MUSIC und ESPRIT finden beide Linien.")
    kind, text = ev.verdict_main(make_analysis(found={"MUSIC": True, "Wurzel-MUSIC": True, "ESPRIT": False, "Periodogramm": True}))
    assert kind == "warning" and text.endswith("MUSIC oder ESPRIT verfehlen mindestens eine Linie.")
    kind, text = ev.verdict_main(make_analysis(found={"MUSIC": False, "Wurzel-MUSIC": True, "ESPRIT": True, "Periodogramm": True}))
    assert kind == "warning"


def test_main_verdict_one_line_with_the_bound_and_with_a_wrong_count():
    s = ev.Settings(lines=1)
    kind, text = ev.verdict_main(make_analysis(settings=s, p=2, k_aic=3, k_mdl=2, err_bins={"ESPRIT": [-0.0123]}, crb=0.0194, resolved=None, true_hz=[200.0]))
    assert kind == "info"
    assert text == "2 Komponenten (1 Linien im Modell), Matrix 21 × 21; Linienzahl nach AIC 1,5, nach MDL 1. Fehler der Frequenz mit ESPRIT 0,012 Bins gegen die Schranke 0,019 Bins."
    kind, text = ev.verdict_main(make_analysis(settings=s, err_bins={"ESPRIT": None}, resolved=None))
    assert kind == "warning" and text.endswith("Die Zahl der geschätzten Frequenzen stimmt nicht mit der Wahrheit überein.")


def test_main_verdict_three_lines():
    s = ev.Settings(lines=3)
    ok = make_analysis(settings=s, p=6, resolved=None)
    assert ev.verdict_main(ok)[0] == "success" and "6 Komponenten (3 Linien im Modell)" in ev.verdict_main(ok)[1]
    bad = make_analysis(settings=s, p=6, resolved=None, found={"MUSIC": False, "Wurzel-MUSIC": False, "ESPRIT": True, "Periodogramm": True})
    assert ev.verdict_main(bad)[0] == "warning"


def test_half_and_decimal_comma():
    assert ev.half(4) == "2" and ev.half(0) == "0" and ev.half(3) == "1,5" and ev.half(7) == "3,5"
    assert ev.de(1.26, 1) == "1,3"
    assert ev.de(3.0, 0) == "3" and ev.de(0.04, 3) == "0,040"


def _res_rows(music=0.995, esprit=1.0, hann=0.2, burg=0.565):
    deltas = [0.5, 1.0, 2.0]
    return [{"label": "MUSIC (m = n/3), gefunden", "deltas": deltas, "shares": [0.1, music, 1.0]}, {"label": "ESPRIT (m = n/3), gefunden", "deltas": deltas, "shares": [0.2, esprit, 1.0]},
            {"label": "Hann-Periodogramm, aufgelöst", "deltas": deltas, "shares": [0.0, hann, 0.5]}, {"label": "Burg p = 16, aufgelöst", "deltas": deltas, "shares": [0.0, burg, 1.0]}]


def test_verdict_resolution_text():
    kind, text = ev.verdict_resolution(_res_rows())
    assert kind == "success"
    assert text == ("Bei 1 Bin Abstand finden MUSIC 100 % und ESPRIT 100 % der Läufe beide Linien, das Hann-Periodogramm trennt sie in 20 %, AR (Burg, Ordnung 16) in 56 %. "
                    "**Das gilt bei bekannter Linienzahl.**")
    assert "MUSIC 90 %" in ev.verdict_resolution(_res_rows(music=0.9))[1] and "AR (Burg, Ordnung 16) in 0 %" in ev.verdict_resolution(_res_rows(burg=0.0))[1]


def _acc_rows():
    def m(r, o=0.0):
        return {"ratio": r, "outliers": o}
    rows = []
    for n in (64, 256):
        for snr in (5.0, 10.0):
            rows.append({"n": n, "snr": snr, "crb": 0.02, "runs": 100,
                         "methods": {"Periodogramm, Rechteck": m(0.92 if n == 256 else 7.0), "Periodogramm, Hann": m(1.4), "MUSIC": m(1.09), "Wurzel-MUSIC": m(1.09), "ESPRIT": m(0.954)}})
    return rows


def test_verdict_accuracy_uses_the_longest_series_at_five_db():
    kind, text = ev.verdict_accuracy(_acc_rows())
    assert kind == "info"
    assert text == ("Bei n = 256 und 5 dB liegt die Frequenzgenauigkeit des Rechteck-Periodogramms beim 0,92-fachen der Schranke, die von MUSIC beim 1,09-fachen, die von ESPRIT beim 0,95-fachen. "
                    "**Für eine einzelne Linie gewinnen die Unterraumverfahren nichts.**")


def _count_rows(mdl=0.98, aic=0.67, over=0.33):
    return [{"lines": 2, "snr": 0.0, "runs": 200, "AIC": {"right": aic, "under": 0.0, "over": over}, "MDL": {"right": mdl, "under": 0.0, "over": 0.02}},
            {"lines": 1, "snr": 0.0, "runs": 200, "AIC": {"right": 0.1, "under": 0.0, "over": 0.9}, "MDL": {"right": 0.1, "under": 0.0, "over": 0.9}}]


def test_verdict_count_picks_two_lines_at_zero_db():
    kind, text = ev.verdict_count(_count_rows())
    assert kind == "warning"
    assert text == ("Bei zwei Linien und 0 dB je Abtastwert treffen MDL die Zahl der Komponenten in 98 % der Läufe, AIC in 67 % (zu groß in 33 %). "
                    "**AIC überschätzt, MDL unterschätzt erst bei sehr schwachen Linien.**")


def _matrix_rows():
    ms = [6, 8, 12, 16, 24]
    f = {6: (0.48, 0.5), 8: (0.9, 0.92), 12: (1.0, 1.0), 16: (1.0, 1.0), 24: (1.0, 1.0)}
    return [{"m": m, "runs": 200, "MUSIC": {"found": f[m][0], "rmse": 0.1}, "ESPRIT": {"found": f[m][1], "rmse": 0.2 if m != 24 else 0.0251}} for m in ms]


def test_verdict_matrix_uses_the_smallest_m_the_value_twelve_and_the_fifth_row():
    kind, text = ev.verdict_matrix(_matrix_rows())
    assert kind == "info"
    assert text == ("Mit m = 6 finden MUSIC 48 % und ESPRIT 50 % der Läufe beide Linien, mit m = 12 100 % und 100 %; ab m = 24 liegt der Fehler bei 0,025 Bins (ESPRIT). "
                    "**Die Matrix muss deutlich größer als die Zahl der Komponenten sein.**")


def test_verdict_coloured_text():
    rows = [{"label": "weißes Rauschen", "MUSIC": 1.0, "ESPRIT": 1.0, "Burg": 1.0}, {"label": "gefärbt, gleiche Leistung", "MUSIC": 0.97, "ESPRIT": 1.0, "Burg": 1.0},
            {"label": "gefärbt, zehnfache Leistung", "MUSIC": 0.755, "ESPRIT": 0.67, "Burg": 0.995}]
    kind, text = ev.verdict_coloured(rows)
    assert kind == "warning"
    assert text == ("Auf einem Hintergrund mit zehnfacher Leistung finden MUSIC 76 % und ESPRIT 67 % der Läufe beide Linien, AR 100 %; bei gleicher Leistung 97 %. "
                    "**Die Annahme „weißes Rauschen“ trägt das Verfahren.**")


def test_verdict_order_text():
    rows = [{"p": 2, "runs": 200, "rmse": 0.1157, "outliers": 0.055, "extra": 0.645}, {"p": 4, "runs": 200, "rmse": 0.0441, "outliers": 0.0, "extra": 0.0},
            {"p": 12, "runs": 200, "rmse": 0.0446, "outliers": 0.0, "extra": 0.02}]
    kind, text = ev.verdict_order(rows)
    assert kind == "warning"
    assert text == ("Mit Ordnung 2 statt 4 liegt der Frequenzfehler bei 0,116 Bins und 64 % der Läufe zeigen einen Zusatzgipfel; mit Ordnung 4 0,044 Bins, mit Ordnung 12 0,045 Bins. "
                    "**Zu kleine Ordnung schadet, zu große kaum.**")


def test_verdict_height_text():
    kind, text = ev.verdict_height({"runs": 100, "true_db": -20.0, "periodogram_db": -19.96, "music_db": -11.93, "music_std_db": 4.16})
    assert kind == "info"
    assert text == ("Eine Linie 20 dB unter der starken liegt im Periodogramm bei -20,0 dB, im MUSIC-Pseudospektrum bei -11,9 dB (Streuung 4,2 dB). "
                    "**Das Pseudospektrum ist kein Leistungsspektrum.**")
