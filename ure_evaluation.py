"""Analyse, Kennzahlen, Experimente (feste Ströme) und Verdikte der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import signal_generator as G
import ure_compare as H
import ure_constants as C
import ure_subspace as U

DB = 10.0 / math.log(10.0)


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def half(k: int) -> str:
    """Komponenten geteilt durch zwei: ganze Zahl oder mit Dezimalkomma (z. B. 1,5)."""
    return str(k // 2) if k % 2 == 0 else de(k / 2.0, 1)


def matrix_size(n: int, choice: str) -> int:
    """Größe m der Kovarianzmatrix: n / Teiler, mindestens 12, höchstens n / 2."""
    return min(max(n // C.M_DIVISOR[choice], C.M_FLOOR), n // 2)


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    lines: int = C.DEFAULT_LINE_COUNT
    line: float = C.DEFAULT_LINE
    gap: float = C.DEFAULT_GAP
    snr: int = C.DEFAULT_SNR
    background: str = C.DEFAULT_BACKGROUND
    m: str = C.DEFAULT_M
    order_mode: str = C.DEFAULT_ORDER_MODE
    order: int = C.DEFAULT_ORDER
    seed: int = C.DEFAULT_SEED

    @property
    def m_size(self) -> int:
        return matrix_size(self.n, self.m)

    @property
    def f1_hz(self) -> float:
        return self.line * C.FS

    @property
    def f2_hz(self) -> float:
        """Lage der Linie 2: oberhalb von Linie 1; läge sie jenseits der Nyquist-Grenze (zwei Bins davor), dann unterhalb im selben Abstand."""
        step = self.gap * C.FS / self.n
        up = self.f1_hz + step
        return up if up <= C.FS / 2 - 2.0 * C.FS / self.n else self.f1_hz - step

    @property
    def f3_hz(self) -> float:
        """Lage der Linie 3: 0,13 f_s über Linie 1, sonst darunter."""
        up = self.f1_hz + C.THIRD_OFFSET * C.FS
        return up if up <= C.FS / 2 - 2.0 * C.FS / self.n else self.f1_hz - C.THIRD_OFFSET * C.FS

    @property
    def line_hz(self) -> list[float]:
        return sorted([self.f1_hz, self.f2_hz, self.f3_hz][:self.lines])


def burg_spectrum(x, order: int, grid) -> np.ndarray:
    """Burg-Schätzung der Ordnung `order` als Spektrum auf dem Frequenzgitter (Bruchteile der Abtastrate)."""
    a, err, _ = H.burg(x, order)
    return H.ar_spectrum(a, float(err[-1]), grid, 1.0)


def amplitude(snr_db: float) -> float:
    return math.sqrt(2.0 * 10.0 ** (snr_db / 10.0))


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    m: int
    n_snap: int
    p: int                          # verwendete Zahl der Komponenten (2 je Linie)
    k_aic: int
    k_mdl: int
    eig: np.ndarray                 # Eigenwerte absteigend
    f_hz: np.ndarray                # Auswertegitter
    x: np.ndarray
    music: np.ndarray               # Pseudospektrum
    fft: np.ndarray                 # Periodogramm, Hann
    burg: np.ndarray                # AR-Spektrum, Ordnung 16 (höchstens n/3)
    burg_order: int
    true_hz: list                   # wahre Frequenzen
    est_hz: dict                    # geschätzte Frequenzen je Verfahren
    err_bins: dict                  # Fehler je Linie in Bins von 1/n (None, wenn die Zahl der Frequenzen nicht stimmt)
    found: dict                     # alle Linien innerhalb des erlaubten Fehlers
    crb: float | None               # Cramér-Rao-Schranke in Bins (nur bei einer Linie)
    resolved: dict | None           # bei zwei Linien: getrennt im Pseudospektrum, im Periodogramm, im AR-Spektrum
    ops: int
    ops_fft: int


def build_signal(s: Settings):
    rng = G.SplitMix64(s.seed)
    ph = 2.0 * math.pi * rng.uniform(3)
    factor = C.BACKGROUND_FACTOR[s.background]
    if factor is None:
        poles, ar_sigma, white = (), 0.0, 1.0
    else:
        poles = C.COLOUR_POLES
        ar_sigma = G.ar_unit_sigma(G.ar_from_poles(poles)) * math.sqrt(factor)
        white = 0.0
    a1 = amplitude(s.snr)
    freqs = [s.f1_hz, s.f2_hz, s.f3_hz][:s.lines]
    lines = [(f, a1, float(ph[i])) for i, f in enumerate(freqs)]
    sig = G.make_signal(s.n, C.FS, lines, poles, ar_sigma, white, rng)
    return sig


def _match(est, true_hz, n) -> list | None:
    """Fehler in Bins je wahrer Linie (aufsteigend sortiert), None, wenn die Zahl der Frequenzen nicht stimmt."""
    if len(est) != len(true_hz):
        return None
    return [float((e - t) * n / C.FS) for e, t in zip(sorted(est), sorted(true_hz))]


def analyse(s: Settings) -> Analysis:
    sig = build_signal(s)
    n, m = s.n, s.m_size
    r = U.covariance(sig.x, m)
    w, v = U.eigendecomposition(r)
    n_snap = n - m + 1
    k_aic, k_mdl = U.estimate_count(w, n_snap, "AIC"), U.estimate_count(w, n_snap, "MDL")
    if s.order_mode == "fest":
        p = s.order
    else:
        k = k_aic if s.order_mode == "AIC" else k_mdl
        p = max(2, k + k % 2)
    p = min(p, m - 2)
    count = max(1, p // 2)
    grid = np.arange(1, C.GRID * n // 2) / (C.GRID * n)
    f_hz = grid * C.FS
    music = U.music_spectrum(v, p, grid)
    wn = H.window("Hann", n)
    fft = np.abs(np.fft.rfft(sig.x * wn, C.GRID * n))[1:C.GRID * n // 2] ** 2
    b_order = min(C.BURG_ORDER, n // 3)
    burg = burg_spectrum(sig.x, b_order, grid)
    true_hz = s.line_hz
    est = {"MUSIC": [f * C.FS for f in U.music_frequencies(v, p, count, grid)], "Wurzel-MUSIC": [f * C.FS for f in U.root_music(v, p)],
           "ESPRIT": [f * C.FS for f in U.esprit(v, p)], "Periodogramm": [f * C.FS for f in U.periodogram_frequencies(sig.x, count, wn)]}
    err = {k: _match(vals, true_hz, n) for k, vals in est.items()}
    if s.lines >= 2:
        spacing = min(b - a_ for a_, b in zip(true_hz, true_hz[1:])) * n / C.FS
        tol = C.FOUND_FRACTION * spacing
    else:
        tol = 0.5
    found = {k: (e is not None and all(abs(x) < tol for x in e)) for k, e in err.items()}
    resolved = None
    if s.lines == 2:
        b1, b2 = s.f1_hz * n / C.FS, s.f2_hz * n / C.FS
        resolved = {"MUSIC": H.lines_resolved(np.r_[0.0, music], C.GRID, b1, b2),
                    "Periodogramm": H.lines_resolved(np.r_[0.0, fft], C.GRID, b1, b2), "AR": H.lines_resolved(np.r_[0.0, burg], C.GRID, b1, b2)}
    crb = U.crb_bins(n, float(s.snr)) if s.lines == 1 else None
    return Analysis(s, m, n_snap, p, k_aic, k_mdl, w, f_hz, sig.x, music, fft, burg, b_order, true_hz, est, err, found, crb, resolved, U.ops_subspace(n, m), n // 2 * int(round(math.log2(n))))


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def _noise_sigma(snr_db: float) -> float:
    """Rauschstandardabweichung für Linien mit Amplitude 1: Leistung 0,5 gegen 10^(SNR/10)."""
    return math.sqrt(0.5 / 10.0 ** (snr_db / 10.0))


def _lines(runs: int, n: int, freqs, snr_db: float, seed: int, phases=None):
    """runs x n Reihen: Summe reiner Linien (Amplitude 1, zufällige Phasen) plus weißes Rauschen. freqs: Liste von Feldern der Länge runs oder Zahlen."""
    rng = G.SplitMix64(seed)
    t = np.arange(n)
    x = np.zeros((runs, n))
    for f in freqs:
        f = np.broadcast_to(np.asarray(f, dtype=float), (runs,))
        ph = 2.0 * math.pi * rng.uniform(runs)
        x += np.cos(2.0 * np.pi * f[:, None] * t[None, :] + ph[:, None])
    return x + _noise_sigma(snr_db) * rng.normal(runs * n).reshape(runs, n), rng


def _two_lines(runs: int, n: int, delta: float, snr_db: float, seed: int):
    rng = G.SplitMix64(seed)
    f1 = (14.0 + 2.0 * rng.uniform(runs)) / n
    f2 = f1 + delta / n
    ph = 2.0 * math.pi * rng.uniform(2 * runs).reshape(2, runs)
    t = np.arange(n)
    x = (np.cos(2.0 * np.pi * f1[:, None] * t[None, :] + ph[0][:, None]) + np.cos(2.0 * np.pi * f2[:, None] * t[None, :] + ph[1][:, None])
         + _noise_sigma(snr_db) * rng.normal(runs * n).reshape(runs, n))
    return x, f1, f2


def _found(est, f1, f2, tol) -> bool:
    return len(est) == 2 and abs(est[0] - f1) < tol and abs(est[1] - f2) < tol


def sweep_resolution(n: int = C.EXP_RES_N, deltas=C.EXP_RES_DELTAS, runs: int = C.EXP_RES_RUNS, snr: float = C.EXP_RES_SNR) -> list[dict]:
    """Zwei gleich starke Linien (SNR je Linie) in weißem Rauschen, Modellordnung 4 bekannt: Anteil der Läufe, in denen beide Frequenzen innerhalb von Δ/4 liegen („gefunden“), und
    Anteil mit zwei getrennten Maxima und einem Einbruch von mindestens 3 dB („aufgelöst“) je Verfahren und Abstand (Bins von 1 / n)."""
    pad = C.GRID
    grid = np.arange(1, pad * n // 2) / (pad * n)
    grid_full = np.arange(pad * n // 2 + 1) / (pad * n)
    labels = ["MUSIC (m = n/3), gefunden", "Wurzel-MUSIC (m = n/3), gefunden", "ESPRIT (m = n/3), gefunden", "MUSIC (m = n/2), gefunden", "ESPRIT (m = n/2), gefunden",
              "MUSIC (m = n/3), aufgelöst", "Burg p = 16, aufgelöst", "Multitaper NW = 2, aufgelöst", "Hann-Periodogramm, aufgelöst"]
    rows = {k: [] for k in labels}
    for d in deltas:
        x, f1, f2 = _two_lines(runs, n, d, snr, C.SWEEP_SEED)
        counts = {k: 0 for k in labels}
        for r in range(runs):
            tol = d / (4.0 * n)
            for div, tag in ((3, "m = n/3"), (2, "m = n/2")):
                m = max(n // div, 12)
                w, v = U.eigendecomposition(U.covariance(x[r], m))
                counts[f"MUSIC ({tag}), gefunden"] += _found(U.music_frequencies(v, 4, 2, grid), f1[r], f2[r], tol)
                counts[f"ESPRIT ({tag}), gefunden"] += _found(U.esprit(v, 4), f1[r], f2[r], tol)
                if div == 3:
                    counts[f"Wurzel-MUSIC ({tag}), gefunden"] += _found(U.root_music(v, 4), f1[r], f2[r], tol)
                    counts["MUSIC (m = n/3), aufgelöst"] += H.lines_resolved(np.r_[0.0, U.music_spectrum(v, 4, grid)], pad, f1[r] * n, f2[r] * n)
            counts["Burg p = 16, aufgelöst"] += H.lines_resolved(burg_spectrum(x[r], 16, grid_full), pad, f1[r] * n, f2[r] * n)
            counts["Multitaper NW = 2, aufgelöst"] += H.lines_resolved(H.multitaper_equal(x[r], H.dpss(n, 2, 3), 1.0, pad * n), pad, f1[r] * n, f2[r] * n)
            counts["Hann-Periodogramm, aufgelöst"] += H.lines_resolved(np.abs(np.fft.rfft(x[r] * H.window("Hann", n), pad * n)) ** 2, pad, f1[r] * n, f2[r] * n)
        for k in labels:
            rows[k].append(float(counts[k] / runs))
    return [{"label": k, "deltas": list(deltas), "shares": rows[k]} for k in labels]


def sweep_accuracy(ns=C.EXP_ACC_NS, snrs=C.EXP_ACC_SNR, runs: int = C.EXP_ACC_RUNS) -> list[dict]:
    """Frequenz einer einzelnen Linie (Matrix n/3): RMSE geteilt durch die Cramér-Rao-Schranke je Verfahren (Ausreißer ab 3 Bins getrennt gezählt); Periodogramm mit Rechteck- und mit Hann-Fenster."""
    rows = []
    for n in ns:
        grid = np.arange(1, 16 * n // 2) / (16 * n)
        for snr in snrs:
            rng = G.SplitMix64(C.SWEEP_SEED + n)
            f0 = (n / 5.0 + rng.uniform(runs)) / n
            x, _ = _lines(runs, n, [f0], snr, C.SWEEP_SEED + n + 1)
            crb = U.crb_bins(n, snr)
            errs = {"Periodogramm, Rechteck": [], "Periodogramm, Hann": [], "MUSIC": [], "Wurzel-MUSIC": [], "ESPRIT": []}
            for r in range(runs):
                errs["Periodogramm, Rechteck"].append((U.periodogram_frequencies(x[r], 1)[0] - f0[r]) * n)
                errs["Periodogramm, Hann"].append((U.periodogram_frequencies(x[r], 1, H.window("Hann", n))[0] - f0[r]) * n)
                w, v = U.eigendecomposition(U.covariance(x[r], n // 3))
                mf = U.music_frequencies(v, 2, 1, grid)
                errs["MUSIC"].append((mf[0] - f0[r]) * n if mf else 1e9)
                for key, fn in (("Wurzel-MUSIC", U.root_music), ("ESPRIT", U.esprit)):
                    try:
                        errs[key].append((fn(v, 2)[0] - f0[r]) * n)
                    except (IndexError, np.linalg.LinAlgError):
                        errs[key].append(1e9)
            row = {"n": n, "snr": snr, "crb": crb, "runs": runs, "methods": {}}
            for key, e in errs.items():
                e = np.array(e)
                ok = np.abs(e) < C.EXP_ACC_OUTLIER
                row["methods"][key] = {"ratio": float(math.sqrt(np.mean(e[ok] ** 2)) / crb) if ok.any() else float("nan"), "outliers": float(1.0 - ok.mean())}
            rows.append(row)
    return rows


def sweep_count(n: int = C.EXP_COUNT_N, m: int = C.EXP_COUNT_M, lines=C.EXP_COUNT_LINES, snrs=C.EXP_COUNT_SNR, runs: int = C.EXP_COUNT_RUNS) -> list[dict]:
    """Zahl der Komponenten (2 je Linie) aus den Eigenwerten mit AIC und MDL: Anteil richtig, zu klein, zu groß je Linienzahl und SNR."""
    rows = []
    for count, freqs in lines:
        for snr in snrs:
            x, _ = _lines(runs, n, list(freqs), snr, C.SWEEP_SEED + 20 + count)
            ks = {"AIC": [], "MDL": []}
            for r in range(runs):
                w, _ = U.eigendecomposition(U.covariance(x[r], m))
                for kind in ks:
                    ks[kind].append(U.estimate_count(w, n - m + 1, kind))
            row = {"lines": count, "snr": snr, "runs": runs}
            for kind, vals in ks.items():
                a = np.array(vals)
                row[kind] = {"right": float(np.mean(a == 2 * count)), "under": float(np.mean(a < 2 * count)), "over": float(np.mean(a > 2 * count))}
            rows.append(row)
    return rows


def sweep_matrix(n: int = C.EXP_MATRIX_N, delta: float = C.EXP_MATRIX_DELTA, ms=C.EXP_MATRIX_MS, runs: int = C.EXP_MATRIX_RUNS, snr: float = C.EXP_RES_SNR) -> list[dict]:
    """Größe m der Kovarianzmatrix: Anteil gefunden und RMSE der Frequenz in Bins (nur gefundene Läufe) für MUSIC und ESPRIT bei zwei Linien im Abstand delta."""
    grid = np.arange(1, C.GRID * n // 2) / (C.GRID * n)
    x, f1, f2 = _two_lines(runs, n, delta, snr, C.SWEEP_SEED + 30)
    rows = []
    for m in ms:
        row = {"m": m, "runs": runs}
        for name, fn in (("MUSIC", lambda v: U.music_frequencies(v, 4, 2, grid)), ("ESPRIT", lambda v: U.esprit(v, 4))):
            ok, err = 0, []
            for r in range(runs):
                w, v = U.eigendecomposition(U.covariance(x[r], m))
                est = fn(v)
                if _found(est, f1[r], f2[r], delta / (4.0 * n)):
                    ok += 1
                    err += [(est[0] - f1[r]) * n, (est[1] - f2[r]) * n]
            row[name] = {"found": ok / runs, "rmse": float(math.sqrt(np.mean(np.square(err)))) if err else float("nan")}
        rows.append(row)
    return rows


def sweep_coloured(n: int = C.EXP_COLOUR_N, m: int = C.EXP_COLOUR_M, freqs=C.EXP_COLOUR_FREQS, runs: int = C.EXP_COLOUR_RUNS, snr: float = C.EXP_RES_SNR) -> list[dict]:
    """Zwei Linien auf weißem und auf gefärbtem Hintergrund (AR-Resonanz; gleiche und zehnfache Leistung gegen das weiße Rauschniveau): Anteil, in dem MUSIC und ESPRIT beide Linien
    (innerhalb 0,01 f_s) finden und Burg (p = 16) beide unter den drei höchsten Gipfeln."""
    grid = np.arange(1, C.GRID * n // 2) / (C.GRID * n)
    sig = G.ar_unit_sigma(G.ar_from_poles(C.COLOUR_POLES))
    t = np.arange(n)
    rows = []
    for label, factor in (("weißes Rauschen", None), ("gefärbt, gleiche Leistung", 1.0), ("gefärbt, zehnfache Leistung", 10.0)):
        rng = G.SplitMix64(C.SWEEP_SEED + 40)
        ph = 2.0 * math.pi * rng.uniform(2 * runs).reshape(2, runs)
        x = sum(np.cos(2.0 * np.pi * f * t[None, :] + ph[i][:, None]) for i, f in enumerate(freqs))
        if factor is None:
            x = x + _noise_sigma(snr) * rng.normal(runs * n).reshape(runs, n)
        else:
            x = x + G.make_noise_batch(runs, n, C.COLOUR_POLES, sig * math.sqrt(factor) * _noise_sigma(snr), 0.0, G.SplitMix64(C.SWEEP_SEED + 41))
        hits = {"MUSIC": 0, "ESPRIT": 0, "Burg": 0}
        for r in range(runs):
            w, v = U.eigendecomposition(U.covariance(x[r], m))
            for name, est in (("MUSIC", U.music_frequencies(v, 4, 2, grid)), ("ESPRIT", U.esprit(v, 4))):
                hits[name] += len(est) == 2 and abs(est[0] - freqs[0]) < 0.01 and abs(est[1] - freqs[1]) < 0.01
            spec = burg_spectrum(x[r], 16, grid)
            top = U.peak_frequencies(spec, grid, 3)
            hits["Burg"] += any(abs(q - freqs[0]) < 0.01 for q in top) and any(abs(q - freqs[1]) < 0.01 for q in top)
        rows.append({"label": label, "runs": runs, **{k: v / runs for k, v in hits.items()}})
    return rows


def sweep_order(n: int = C.EXP_ORDER_N, m: int = C.EXP_ORDER_M, freqs=C.EXP_ORDER_FREQS, ps=C.EXP_ORDER_PS, runs: int = C.EXP_ORDER_RUNS, snr: float = C.EXP_RES_SNR) -> list[dict]:
    """Falsche Modellordnung p bei zwei Linien (wahr: 4): RMSE der beiden höchsten Gipfel (Bins, ohne Ausreißer ab 3 Bins), Anteil der Ausreißer und der Läufe mit einem
    Zusatzgipfel über 10 % der höchsten Höhe."""
    grid = np.arange(1, C.GRID * n // 2) / (C.GRID * n)
    x, _ = _lines(runs, n, list(freqs), snr, C.SWEEP_SEED + 50)
    rows = []
    for p in ps:
        err, extra = [], 0
        for r in range(runs):
            w, v = U.eigendecomposition(U.covariance(x[r], m))
            spec = U.music_spectrum(v, p, grid)
            allpk = U.local_maxima(spec)
            top = sorted(allpk, key=lambda i: -spec[i])
            extra += any(min(abs(grid[i] - freqs[0]), abs(grid[i] - freqs[1])) >= 0.01 and spec[i] > 0.1 * spec.max() for i in top[:20])
            pk = U.peak_frequencies(spec, grid, 2)
            if len(pk) == 2:
                err += [(pk[0] - freqs[0]) * n, (pk[1] - freqs[1]) * n]
        e = np.array(err)
        ok = np.abs(e) < C.EXP_ACC_OUTLIER
        rows.append({"p": p, "runs": runs, "rmse": float(math.sqrt(np.mean(e[ok] ** 2))), "outliers": float(1.0 - ok.mean()), "extra": extra / runs})
    return rows


def sweep_height(n: int = C.EXP_HEIGHT_N, m: int = C.EXP_HEIGHT_M, runs: int = C.EXP_HEIGHT_RUNS) -> dict:
    """Zwei Linien 20 dB auseinander (stark: 20 dB SNR): Höhe der schwachen gegen die starke Linie im Periodogramm (Hann) und im MUSIC-Pseudospektrum, in dB."""
    f1, f2 = 0.13, 0.31
    rng = G.SplitMix64(C.SWEEP_SEED + 60)
    t = np.arange(n)
    ph = 2.0 * math.pi * rng.uniform(2 * runs).reshape(2, runs)
    x = np.cos(2.0 * np.pi * f1 * t[None, :] + ph[0][:, None]) + 0.1 * np.cos(2.0 * np.pi * f2 * t[None, :] + ph[1][:, None]) + _noise_sigma(20.0) * rng.normal(runs * n).reshape(runs, n)
    grid = np.arange(1, C.GRID * n // 2) / (C.GRID * n)
    i1, i2 = int(round(f1 * C.GRID * n)) - 1, int(round(f2 * C.GRID * n)) - 1
    music, pgram = [], []
    for r in range(runs):
        w, v = U.eigendecomposition(U.covariance(x[r], m))
        spec = U.music_spectrum(v, 4, grid)
        music.append(DB * math.log(spec[i2 - 3:i2 + 4].max() / spec[i1 - 3:i1 + 4].max()))
        pg = np.abs(np.fft.rfft(x[r] * H.window("Hann", n), C.GRID * n)[1:C.GRID * n // 2]) ** 2
        pgram.append(DB * math.log(pg[i2 - 2:i2 + 5].max() / pg[i1 - 2:i1 + 5].max()))
    return {"runs": runs, "true_db": -20.0, "periodogram_db": float(np.mean(pgram)), "music_db": float(np.mean(music)), "music_std_db": float(np.std(music))}


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def verdict_main(a: Analysis) -> tuple[str, str]:
    s = a.settings
    base = f"{a.p} Komponenten ({a.p // 2} Linien im Modell), Matrix {a.m} × {a.m}; Linienzahl nach AIC {half(a.k_aic)}, nach MDL {half(a.k_mdl)}."
    if s.lines == 1:
        e = a.err_bins["ESPRIT"]
        if e is None:
            return "warning", base + " Die Zahl der geschätzten Frequenzen stimmt nicht mit der Wahrheit überein."
        return "info", base + f" Fehler der Frequenz mit ESPRIT {de(abs(e[0]), 3)} Bins gegen die Schranke {de(a.crb, 3)} Bins."
    if s.lines == 2 and a.resolved is not None:
        if a.found["ESPRIT"] and a.found["MUSIC"]:
            return "success", base + " MUSIC und ESPRIT finden beide Linien" + ("; das Periodogramm trennt sie nicht." if not a.found["Periodogramm"] else ".")
        return "warning", base + " MUSIC oder ESPRIT verfehlen mindestens eine Linie."
    ok = all(a.found[k] for k in ("MUSIC", "ESPRIT"))
    return ("success" if ok else "warning"), base + (" MUSIC und ESPRIT finden alle Linien." if ok else " MUSIC oder ESPRIT verfehlen mindestens eine Linie.")


def verdict_resolution(rows: list[dict]) -> tuple[str, str]:
    by = {r["label"]: dict(zip(r["deltas"], r["shares"])) for r in rows}
    return "success", (f"Bei 1 Bin Abstand finden MUSIC {de(100 * by['MUSIC (m = n/3), gefunden'][1.0], 0)} % und ESPRIT {de(100 * by['ESPRIT (m = n/3), gefunden'][1.0], 0)} % der Läufe "
                       f"beide Linien, das Hann-Periodogramm trennt sie in {de(100 * by['Hann-Periodogramm, aufgelöst'][1.0], 0)} %, AR (Burg, Ordnung 16) in "
                       f"{de(100 * by['Burg p = 16, aufgelöst'][1.0], 0)} %. **Das gilt bei bekannter Linienzahl.**")


def verdict_accuracy(rows: list[dict]) -> tuple[str, str]:
    r = [x for x in rows if x["n"] == max(y["n"] for y in rows) and x["snr"] == 5.0][0]
    m = r["methods"]
    return "info", (f"Bei n = {r['n']} und 5 dB liegt die Frequenzgenauigkeit des Rechteck-Periodogramms beim {de(m['Periodogramm, Rechteck']['ratio'], 2)}-fachen der Schranke, "
                    f"die von MUSIC beim {de(m['MUSIC']['ratio'], 2)}-fachen, die von ESPRIT beim {de(m['ESPRIT']['ratio'], 2)}-fachen. **Für eine einzelne Linie gewinnen die "
                    "Unterraumverfahren nichts.**")


def verdict_count(rows: list[dict]) -> tuple[str, str]:
    r = [x for x in rows if x["lines"] == 2 and x["snr"] == 0.0][0]
    return "warning", (f"Bei zwei Linien und 0 dB je Abtastwert treffen MDL die Zahl der Komponenten in {de(100 * r['MDL']['right'], 0)} % der Läufe, AIC in {de(100 * r['AIC']['right'], 0)} % "
                       f"(zu groß in {de(100 * r['AIC']['over'], 0)} %). **AIC überschätzt, MDL unterschätzt erst bei sehr schwachen Linien.**")


def verdict_matrix(rows: list[dict]) -> tuple[str, str]:
    small = rows[0]
    r12 = [x for x in rows if x["m"] == 12][0]
    return "info", (f"Mit m = {small['m']} finden MUSIC {de(100 * small['MUSIC']['found'], 0)} % und ESPRIT {de(100 * small['ESPRIT']['found'], 0)} % der Läufe beide Linien, mit m = 12 "
                    f"{de(100 * r12['MUSIC']['found'], 0)} % und {de(100 * r12['ESPRIT']['found'], 0)} %; ab m = 24 liegt der Fehler bei {de(rows[4]['ESPRIT']['rmse'], 3)} Bins (ESPRIT). "
                    "**Die Matrix muss deutlich größer als die Zahl der Komponenten sein.**")


def verdict_coloured(rows: list[dict]) -> tuple[str, str]:
    by = {r["label"]: r for r in rows}
    r = by["gefärbt, zehnfache Leistung"]
    return "warning", (f"Auf einem Hintergrund mit zehnfacher Leistung finden MUSIC {de(100 * r['MUSIC'], 0)} % und ESPRIT {de(100 * r['ESPRIT'], 0)} % der Läufe beide Linien, AR {de(100 * r['Burg'], 0)} %; "
                       f"bei gleicher Leistung {de(100 * by['gefärbt, gleiche Leistung']['MUSIC'], 0)} %. **Die Annahme „weißes Rauschen“ trägt das Verfahren.**")


def verdict_order(rows: list[dict]) -> tuple[str, str]:
    by = {r["p"]: r for r in rows}
    return "warning", (f"Mit Ordnung 2 statt 4 liegt der Frequenzfehler bei {de(by[2]['rmse'], 3)} Bins und {de(100 * by[2]['extra'], 0)} % der Läufe zeigen einen Zusatzgipfel; mit Ordnung 4 "
                       f"{de(by[4]['rmse'], 3)} Bins, mit Ordnung 12 {de(by[12]['rmse'], 3)} Bins. **Zu kleine Ordnung schadet, zu große kaum.**")


def verdict_height(res: dict) -> tuple[str, str]:
    return "info", (f"Eine Linie {de(abs(res['true_db']), 0)} dB unter der starken liegt im Periodogramm bei {de(res['periodogram_db'], 1)} dB, im MUSIC-Pseudospektrum bei {de(res['music_db'], 1)} dB "
                    f"(Streuung {de(res['music_std_db'], 1)} dB). **Das Pseudospektrum ist kein Leistungsspektrum.**")
