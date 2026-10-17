"""Kern: Unterraumverfahren für reelle Linien in Rauschen (Kovarianzmatrix, Eigenzerlegung, MUSIC, Wurzel-MUSIC, ESPRIT, Linienzahl nach Wax und Kailath), Periodogrammspitzen
und die Cramér-Rao-Schranke.

Reine Funktionen, kein Zufall, kein Streamlit; nur numpy. Frequenzen sind Bruchteile der Abtastrate. Eine reelle Linie belegt zwei Komponenten (±f): die Modellordnung p ist 2 mal die Linienzahl.
"""

import math

import numpy as np


def snapshots(x, m: int) -> np.ndarray:
    """Überlappende Schnappschüsse x[t .. t + m - 1] als Zeilen: (n - m + 1) x m."""
    x = np.asarray(x, dtype=float)
    if m < 2 or m > len(x):
        raise ValueError("Matrixgröße außerhalb von 2 .. n")
    return np.lib.stride_tricks.sliding_window_view(x, m)


def covariance(x, m: int, forward_backward: bool = True) -> np.ndarray:
    """Stichproben-Kovarianzmatrix (m x m) der Schnappschüsse; mit Vorwärts-Rückwärts-Mittel (R + J R J) / 2 (J spiegelt die Reihenfolge)."""
    s = snapshots(x, m)
    r = s.T @ s / len(s)
    if forward_backward:
        r = 0.5 * (r + r[::-1, ::-1])
    return r


def eigendecomposition(r):
    """Eigenwerte absteigend und die zugehörigen Eigenvektoren (Spalten) einer symmetrischen Matrix."""
    w, v = np.linalg.eigh(r)
    return w[::-1], v[:, ::-1]


def music_spectrum(v, p: int, f) -> np.ndarray:
    """MUSIC-Pseudospektrum 1 / ||E_n^T a(f)||^2 mit dem Rauschunterraum E_n (Eigenvektoren ab Nummer p) und a(f) = exp(-2 pi i f (0 .. m - 1))."""
    m = v.shape[0]
    if not 0 < p < m:
        raise ValueError("Modellordnung außerhalb von 1 .. m - 1")
    en = v[:, p:]
    a = np.exp(-2j * np.pi * np.outer(np.arange(m), np.asarray(f, dtype=float)))
    return 1.0 / np.sum(np.abs(en.T @ a) ** 2, axis=0)


def local_maxima(spec) -> list[int]:
    """Indizes innerer Punkte, die höher als der linke und mindestens so hoch wie der rechte Nachbar sind."""
    v = np.asarray(spec, dtype=float)
    return [i for i in range(1, len(v) - 1) if v[i] > v[i - 1] and v[i] >= v[i + 1]]


def peak_frequencies(spec, f, count: int) -> list[float]:
    """Frequenzen der `count` höchsten lokalen Maxima, aufsteigend sortiert."""
    idx = sorted(local_maxima(spec), key=lambda i: -spec[i])[:count]
    return sorted(float(f[i]) for i in idx)


def music_frequencies(v, p: int, count: int, grid, fine_points: int = 601) -> list[float]:
    """MUSIC-Frequenzen: grobe Suche auf dem Gitter, dann je Gipfel eine feine Suche (fine_points Punkte, ± 1,5 Gitterschritte), damit das Gitter die Genauigkeit nicht begrenzt."""
    grid = np.asarray(grid, dtype=float)
    spec = music_spectrum(v, p, grid)
    step = float(grid[1] - grid[0])
    out = []
    for i in sorted(local_maxima(spec), key=lambda i: -spec[i])[:count]:
        fine = grid[i] + np.linspace(-1.5, 1.5, fine_points) * step
        out.append(float(fine[int(np.argmax(music_spectrum(v, p, fine)))]))
    return sorted(out)


def root_music(v, p: int) -> list[float]:
    """Wurzel-MUSIC: Nullstellen des Polynoms sum_k c_k z^k mit den Diagonalsummen c_k von E_n E_n^T; die p Wurzeln im Einheitskreis, die dem Kreis am nächsten liegen, ergeben die
    Frequenzen (jede Linie einmal, da die Wurzeln zu ±f paarweise auftreten)."""
    m = v.shape[0]
    en = v[:, p:]
    c = en @ en.T
    coef = np.array([np.trace(c, offset=k) for k in range(-(m - 1), m)])
    roots = np.roots(coef[::-1])
    roots = roots[np.abs(roots) <= 1.0]
    roots = roots[np.argsort(np.abs(np.abs(roots) - 1.0))][:p]
    return [float(f) for f in np.sort(np.abs(np.angle(roots)) / (2.0 * np.pi))[::2]]


def esprit(v, p: int) -> list[float]:
    """ESPRIT (Kleinste Quadrate): Die Eigenwerte von Phi = pinv(U_1) U_2 mit den um eine Zeile verschobenen Signalvektoren U liegen bei exp(2 pi i f); jede Linie einmal."""
    us = v[:, :p]
    phi, *_ = np.linalg.lstsq(us[:-1], us[1:], rcond=None)
    f = np.sort(np.abs(np.angle(np.linalg.eigvals(phi))) / (2.0 * np.pi))
    return [float(x) for x in f[::2]]


def estimate_count(eigenvalues, n_snapshots: int, kind: str) -> int:
    """Zahl der Signalkomponenten aus den Eigenwerten (Wax und Kailath): k minimiert -N (m - k) ln(g / a) + Strafe mit dem geometrischen Mittel g und dem arithmetischen
    Mittel a der m - k kleinsten Eigenwerte. MDL: Strafe 1/2 k (2 m - k) ln N; AIC: das Doppelte der Zielfunktion plus 2 k (2 m - k)."""
    w = np.asarray(eigenvalues, dtype=float)
    m = len(w)
    best_k, best = 0, None
    for k in range(0, m - 1):
        tail = np.maximum(w[k:], 1e-300)
        ll = -n_snapshots * (m - k) * math.log(math.exp(float(np.mean(np.log(tail)))) / float(np.mean(tail)))
        crit = ll + 0.5 * k * (2 * m - k) * math.log(n_snapshots) if kind == "MDL" else 2.0 * ll + 2.0 * k * (2 * m - k)
        if best is None or crit < best:
            best_k, best = k, crit
    return best_k


def refine_parabola(spec, i: int, step: float) -> float:
    """Frequenz des Scheitels einer Parabel durch den Logarithmus der drei Werte um das Maximum i (Schrittweite step)."""
    a, b, c = np.log(spec[i - 1]), np.log(spec[i]), np.log(spec[i + 1])
    den = a - 2.0 * b + c
    return (i + (0.5 * (a - c) / den if den != 0 else 0.0)) * step


def periodogram_frequencies(x, count: int, window=None, factor: int = 16) -> list[float]:
    """Frequenzen der `count` höchsten Periodogrammspitzen (Fenster oder Rechteck, factor-fach aufgefüllt, Parabel-Nachschärfung); die ersten und letzten 8 Punkte zählen nicht."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    xw = x if window is None else x * np.asarray(window, dtype=float)
    pg = np.abs(np.fft.rfft(xw, factor * n)) ** 2
    idx = [i for i in local_maxima(pg) if 8 <= i < len(pg) - 8]
    idx = sorted(idx, key=lambda i: -pg[i])[:count]
    return sorted(refine_parabola(pg, i, 1.0 / (factor * n)) for i in idx)


def crb_bins(n: int, snr_db: float) -> float:
    """Cramér-Rao-Schranke der Standardabweichung der Frequenz einer reellen Linie in Bins: sqrt(3 / (pi^2 snr n (n^2 - 1))) * n,
    snr = A^2 / (2 sigma^2) je Abtastwert (Rife und Boorstyn 1974, reelle Linie fern von 0 und Nyquist)."""
    snr = 10.0 ** (snr_db / 10.0)
    return math.sqrt(3.0 / (math.pi ** 2 * snr * n * (n * n - 1.0))) * n


def ops_subspace(n: int, m: int) -> int:
    """Grobe Zählung der Multiplikationen: Kovarianz (n - m + 1) m^2 plus Eigenzerlegung 9 m^3."""
    return (n - m + 1) * m * m + 9 * m ** 3
