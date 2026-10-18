"""Konstanten: Regler-Grenzen, feste Annahmen, Hintergründe, Experimente, Farben und Presets (Kern: ure_subspace.py, ure_compare.py)."""

FS = 1000.0                                  # Abtastrate in Hz (feste Annahme)
N_CHOICES = (32, 64, 128, 256, 512)
DEFAULT_N = 64
LINE_COUNTS = (1, 2, 3)
DEFAULT_LINE_COUNT = 2
LINE_MIN, LINE_MAX, LINE_STEP, DEFAULT_LINE = 0.05, 0.43, 0.0005, 0.2          # Linie 1 als Bruchteil der Abtastrate
GAP_MIN, GAP_MAX, GAP_STEP, DEFAULT_GAP = 0.5, 6.0, 0.25, 2.0                  # Abstand der Linie 2 in Bins von f_s / n
THIRD_OFFSET = 0.13                                                             # Linie 3 liegt 0,13 f_s über (sonst unter) Linie 1
SNR_MIN, SNR_MAX, DEFAULT_SNR = -10, 30, 10                                     # je Linie gegen die Rauschvarianz 1, in dB je Abtastwert
SEED_MAX, DEFAULT_SEED = 999999, 39
BACKGROUNDS = ("weiß", "gefärbt, gleiche Leistung", "gefärbt, zehnfache Leistung")
BACKGROUND_FACTOR = {"weiß": None, "gefärbt, gleiche Leistung": 1.0, "gefärbt, zehnfache Leistung": 10.0}
DEFAULT_BACKGROUND = "weiß"
COLOUR_POLES = ((0.2, 0.97),)                                                   # Resonanz des gefärbten Hintergrunds
M_CHOICES = ("n/6", "n/4", "n/3", "n/2")
M_DIVISOR = {"n/6": 6, "n/4": 4, "n/3": 3, "n/2": 2}
DEFAULT_M = "n/3"
M_FLOOR = 12                                                                    # kleinste Matrixgröße
ORDER_MODES = ("fest", "AIC", "MDL")
DEFAULT_ORDER_MODE = "fest"
ORDER_CHOICES = (2, 4, 6, 8)                                                    # Komponenten, 2 je reelle Linie
DEFAULT_ORDER = 4
GRID = 8                                                                        # Gitterpunkte je Bin der Länge n (Darstellung und grobe MUSIC-Suche)
BURG_ORDER = 16
FOUND_FRACTION = 0.25                                                           # „gefunden“: Fehler unter diesem Bruchteil des Linienabstands

# Experimente (feste Ströme; Linien mit Amplitude 1, Rauschvarianz 0,5 / 10^(SNR/10))
SWEEP_SEED = 600000
EXP_RES_N = 64
EXP_RES_DELTAS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
EXP_RES_RUNS = 200
EXP_RES_SNR = 10.0
EXP_ACC_NS = (64, 256)
EXP_ACC_SNR = (-6.0, -3.0, 0.0, 5.0, 10.0, 20.0)
EXP_ACC_RUNS = 100
EXP_ACC_OUTLIER = 3.0                                                           # Fehler ab 3 Bins zählt als Ausreißer
EXP_COUNT_N = 128
EXP_COUNT_M = 32
EXP_COUNT_LINES = ((1, (0.2,)), (2, (0.2, 0.29)), (3, (0.1, 0.2, 0.33)))
EXP_COUNT_SNR = (-9.0, -6.0, -3.0, 0.0, 5.0, 10.0)
EXP_COUNT_RUNS = 200
EXP_MATRIX_N = 64
EXP_MATRIX_DELTA = 2.0
EXP_MATRIX_MS = (6, 8, 12, 16, 24, 32, 48)
EXP_MATRIX_RUNS = 200
EXP_COLOUR_N = 128
EXP_COLOUR_M = 40
EXP_COLOUR_FREQS = (0.1, 0.33)
EXP_COLOUR_RUNS = 200
EXP_ORDER_N = 128
EXP_ORDER_M = 40
EXP_ORDER_FREQS = (0.13, 0.31)
EXP_ORDER_PS = (2, 3, 4, 5, 6, 8, 12)
EXP_ORDER_RUNS = 200
EXP_HEIGHT_N = 128
EXP_HEIGHT_M = 40
EXP_HEIGHT_RUNS = 100

COLOR_MUSIC = "rgba(76,120,168,0.95)"
COLOR_FFT = "rgba(150,150,150,0.7)"
COLOR_BURG = "#E8833A"
COLOR_TRUE = "#B03A2E"
COLOR_EIG = "#7B5EA7"

PRESETS = {
    "Zwei Linien: 1 Bin Abstand": {"n": 64, "lines": 2, "gap": 1.0, "line": 0.2, "snr": 10, "background": "weiß", "m": "n/3", "order_mode": "fest", "order": 4, "seed": 0},
    "Linienzahl: AIC gegen MDL": {"n": 128, "lines": 2, "gap": 3.0, "line": 0.2, "snr": 5, "background": "weiß", "m": "n/4", "order_mode": "AIC", "order": 4, "seed": 10},
    "Eine Linie: nahe der Schranke": {"n": 256, "lines": 1, "gap": 2.0, "line": 0.2, "snr": 5, "background": "weiß", "m": "n/3", "order_mode": "fest", "order": 2, "seed": DEFAULT_SEED},
    "Gefärbtes Rauschen": {"n": 128, "lines": 2, "gap": 3.0, "line": 0.1, "snr": 10, "background": "gefärbt, zehnfache Leistung", "m": "n/3", "order_mode": "fest", "order": 4, "seed": 8},
    "Zu kleine Ordnung": {"n": 128, "lines": 2, "gap": 3.0, "line": 0.13, "snr": 10, "background": "weiß", "m": "n/3", "order_mode": "fest", "order": 2, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Zwei Linien: 1 Bin Abstand": "Zwei gleich starke Linien genau 1 Bin auseinander, n = 64: MUSIC und ESPRIT finden beide; das Hann-Periodogramm zeigt nur einen Gipfel.",
    "Linienzahl: AIC gegen MDL": "Zwei Linien, Ordnung aus den Eigenwerten: MDL trifft die vier Komponenten, AIC schlägt mehr vor und fügt dem Modell Rauschkomponenten hinzu; die Frequenzen passen dann nicht mehr.",
    "Eine Linie: nahe der Schranke": "Eine einzelne Linie: Alle Verfahren liegen dicht an der Cramér-Rao-Schranke; das Rechteck-Periodogramm ist dabei nicht schlechter als die Unterraumverfahren, das Hann-Periodogramm liegt etwas darüber.",
    "Gefärbtes Rauschen": "Hintergrund mit Resonanz und zehnfacher Leistung statt weißem Rauschen: Die Annahme „weiß“ bricht, MUSIC und ESPRIT verfehlen die Linien öfter.",
    "Zu kleine Ordnung": "Zwei Linien (vier Komponenten), aber Ordnung 2: Das Modell hat nur Platz für eine Linie, die andere fehlt in jeder Schätzung.",
}
