"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import ure_constants as C
from ure_evaluation import Analysis

EPS = 1e-30
METHOD_COLORS = {"Periodogramm": C.COLOR_FFT, "MUSIC": C.COLOR_MUSIC, "Wurzel-MUSIC": "#4C9F70", "ESPRIT": "#B58B2E"}


def _rel_db(v):
    """Spektrum in dB gegen das eigene Maximum (ein Pseudospektrum hat keinen Leistungsmaßstab)."""
    v = np.maximum(np.asarray(v, dtype=float), EPS)
    return 10.0 * np.log10(v / v.max())


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_spectrum(a: Analysis, zoom: bool = False) -> go.Figure:
    """Pseudospektrum, Hann-Periodogramm und AR-Spektrum (je gegen das eigene Maximum) über der Frequenz; zoom: Ausschnitt um die wahren Linien."""
    lines = list(a.true_hz)
    if zoom:
        pad = max(4.0 * C.FS / a.settings.n, 0.5 * (max(lines) - min(lines)))
        lo, hi = max(0.0, min(lines) - pad), min(C.FS / 2, max(lines) + pad)
    else:
        lo, hi = 0.0, C.FS / 2
    sel = (a.f_hz >= lo) & (a.f_hz <= hi)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_rel_db(a.fft)[sel], mode="lines", line=dict(color=C.COLOR_FFT, width=1.2), name="Periodogramm, Hann",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_rel_db(a.burg)[sel], mode="lines", line=dict(color=C.COLOR_BURG, width=1.2), name=f"AR (Burg, Ordnung {a.burg_order})",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_rel_db(a.music)[sel], mode="lines", line=dict(color=C.COLOR_MUSIC, width=2.5), name=f"MUSIC-Pseudospektrum (p = {a.p})",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    for name in ("ESPRIT",):
        pts = [f for f in a.est_hz[name] if lo <= f <= hi]
        fig.add_trace(go.Scatter(x=pts, y=[0.0] * len(pts), mode="markers", marker=dict(size=9, color=METHOD_COLORS[name], symbol="diamond"), name=f"{name}-Frequenzen",
                                 hovertemplate=name + " %{x:.2f} Hz<extra></extra>"))
    for pos in lines:
        if lo <= pos <= hi:
            fig.add_vline(x=pos, line=dict(color=C.COLOR_TRUE, width=1.2, dash="dash"))
    fig.update_xaxes(title_text="Frequenz (Hz)", range=[lo, hi])
    fig.update_yaxes(title_text="gegen das eigene Maximum (dB)", range=[-60, 5])
    return _fixed(fig, height=400)


def build_eigenvalues(a: Analysis) -> go.Figure:
    """Eigenwerte der Kovarianzmatrix in dB; die gewählte Zahl der Signalkomponenten trennt die Balken."""
    w = np.maximum(a.eig, EPS)
    colors = [C.COLOR_EIG if i < a.p else "rgba(150,150,150,0.7)" for i in range(len(w))]
    fig = go.Figure(go.Bar(x=list(range(1, len(w) + 1)), y=10.0 * np.log10(w), marker_color=colors, hovertemplate="Eigenwert %{x}: %{y:.1f} dB<extra></extra>"))
    fig.add_vline(x=a.p + 0.5, line=dict(color=C.COLOR_TRUE, width=1.2, dash="dash"))
    fig.update_xaxes(title_text="Nummer des Eigenwerts", dtick=max(1, len(w) // 10))
    fig.update_yaxes(title_text="Eigenwert (dB)")
    fig.update_layout(showlegend=False)
    return _fixed(fig, height=340)


def build_errors(a: Analysis) -> go.Figure:
    """Fehler der Frequenzschätzung in Bins von 1/n je wahrer Linie und Verfahren (nur bei richtiger Zahl geschätzter Frequenzen)."""
    fig = go.Figure()
    labels = [f"{f:.1f} Hz" for f in a.true_hz]
    for name, err in a.err_bins.items():
        if err is None:
            continue
        fig.add_trace(go.Bar(x=labels, y=err, name=name, marker_color=METHOD_COLORS[name], hovertemplate=name + ", %{x}: %{y:.3f} Bins<extra></extra>"))
    if a.crb is not None:
        fig.add_hline(y=a.crb, line=dict(color=C.COLOR_TRUE, dash="dash"))
        fig.add_hline(y=-a.crb, line=dict(color=C.COLOR_TRUE, dash="dash"))
    if not fig.data:
        fig.add_annotation(text="Die Zahl der geschätzten Frequenzen stimmt bei keinem Verfahren<br>mit der Zahl der Linien überein: kein Fehler zu zeigen.", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(title_text="wahre Linie", type="category")
    fig.update_yaxes(title_text="Fehler (Bins von 1/n)")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=340)


def build_resolution(rows) -> go.Figure:
    fig = go.Figure()
    for r in rows:
        dash = "solid" if r["label"].endswith("gefunden") else "dot"
        fig.add_trace(go.Scatter(x=r["deltas"], y=[100 * v for v in r["shares"]], mode="lines+markers", line=dict(dash=dash), name=r["label"],
                                 hovertemplate=r["label"] + ", %{x} Bins: %{y:.0f} %<extra></extra>"))
    fig.update_xaxes(title_text="Abstand der Linien (Bins von f_s/n)")
    fig.update_yaxes(title_text="Anteil der Läufe (%)", range=[-3, 103])
    return _fixed(fig, height=460)


def build_accuracy(rows) -> go.Figure:
    fig = go.Figure()
    for n in dict.fromkeys(r["n"] for r in rows):
        sel = [r for r in rows if r["n"] == n]
        for i, key in enumerate(sel[0]["methods"]):
            fig.add_trace(go.Scatter(x=[r["snr"] for r in sel], y=[r["methods"][key]["ratio"] for r in sel], mode="lines+markers", name=f"{key}, n = {n}",
                                     line=dict(dash="solid" if n == max(r["n"] for r in rows) else "dot", color=["#7f7f7f", "#bcbd22", METHOD_COLORS["MUSIC"], METHOD_COLORS["Wurzel-MUSIC"], METHOD_COLORS["ESPRIT"]][i]),
                                     hovertemplate=f"{key}, n = {n}, %{{x}} dB: %{{y:.2f}}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_TRUE, dash="dash"))
    fig.update_xaxes(title_text="Linie gegen Rauschen (dB je Abtastwert)")
    fig.update_yaxes(title_text="RMSE geteilt durch die Cramér-Rao-Schranke", type="log")
    return _fixed(fig, height=460)


def build_count(rows) -> go.Figure:
    snrs = sorted({r["snr"] for r in rows})
    fig = go.Figure()
    for kind, color in (("AIC", "#B58B2E"), ("MDL", C.COLOR_MUSIC)):
        for count in sorted({r["lines"] for r in rows}):
            sel = [r for r in rows if r["lines"] == count]
            fig.add_trace(go.Scatter(x=snrs, y=[100 * r[kind]["right"] for r in sel], mode="lines+markers", name=f"{kind}, {count} Linie(n)",
                                     line=dict(color=color, dash=["solid", "dash", "dot"][count - 1]), hovertemplate=f"{kind}, {count} Linie(n), %{{x}} dB: %{{y:.0f}} %<extra></extra>"))
    fig.update_xaxes(title_text="Linie gegen Rauschen (dB je Abtastwert)")
    fig.update_yaxes(title_text="Zahl der Komponenten richtig (%)", range=[-3, 103])
    return _fixed(fig, height=420)


def build_matrix(rows) -> go.Figure:
    ms = [r["m"] for r in rows]
    fig = go.Figure()
    for name in ("MUSIC", "ESPRIT"):
        fig.add_trace(go.Scatter(x=ms, y=[100 * r[name]["found"] for r in rows], mode="lines+markers", name=f"{name}: beide Linien gefunden (%)", line=dict(color=METHOD_COLORS[name]),
                                 hovertemplate=name + ", m = %{x}: %{y:.0f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=ms, y=[r["ESPRIT"]["rmse"] for r in rows], mode="lines+markers", name="ESPRIT: Frequenzfehler (Bins)", yaxis="y2", line=dict(color="gray", dash="dot"),
                             hovertemplate="m = %{x}: %{y:.3f} Bins<extra></extra>"))
    fig.update_xaxes(title_text="Größe m der Kovarianzmatrix", type="category")
    fig.update_yaxes(title_text="Läufe mit beiden Linien (%)", range=[-3, 103])
    fig.update_layout(yaxis2=dict(title="RMSE (Bins)", overlaying="y", side="right", rangemode="tozero", fixedrange=True))
    return _fixed(fig, height=380)


def build_coloured(rows) -> go.Figure:
    fig = go.Figure()
    for name, color in (("MUSIC", METHOD_COLORS["MUSIC"]), ("ESPRIT", METHOD_COLORS["ESPRIT"]), ("Burg", C.COLOR_BURG)):
        fig.add_trace(go.Bar(x=[r["label"] for r in rows], y=[100 * r[name] for r in rows], name=name if name != "Burg" else "AR (Burg, p = 16)", marker_color=color,
                             hovertemplate=name + ", %{x}: %{y:.0f} %<extra></extra>"))
    fig.update_yaxes(title_text="beide Linien gefunden (%)", range=[0, 105])
    fig.update_layout(barmode="group")
    return _fixed(fig, height=360)


def build_order(rows) -> go.Figure:
    ps = [str(r["p"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ps, y=[100 * r["extra"] for r in rows], name="Läufe mit Zusatzgipfel (%)", marker_color="rgba(150,150,150,0.7)", hovertemplate="p = %{x}: %{y:.0f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=ps, y=[r["rmse"] for r in rows], mode="lines+markers", name="Frequenzfehler RMSE (Bins)", yaxis="y2", line=dict(color=C.COLOR_MUSIC),
                             hovertemplate="p = %{x}: %{y:.3f} Bins<extra></extra>"))
    fig.update_xaxes(title_text="gewählte Modellordnung p (wahr: 4)", type="category")
    fig.update_yaxes(title_text="Läufe mit Zusatzgipfel (%)", range=[0, 100])
    fig.update_layout(yaxis2=dict(title="RMSE (Bins)", overlaying="y", side="right", rangemode="tozero", fixedrange=True))
    return _fixed(fig, height=360)


def build_height(res: dict) -> go.Figure:
    names = ["wahr", "Periodogramm (Hann)", "MUSIC-Pseudospektrum"]
    vals = [res["true_db"], res["periodogram_db"], res["music_db"]]
    err = [0.0, 0.0, res["music_std_db"]]
    fig = go.Figure(go.Bar(x=names, y=vals, error_y=dict(type="data", array=err), marker_color=["rgba(150,150,150,0.7)", C.COLOR_FFT, C.COLOR_MUSIC], hovertemplate="%{x}: %{y:.1f} dB<extra></extra>"))
    fig.update_yaxes(title_text="Höhe der schwachen gegen die starke Linie (dB)")
    fig.update_layout(showlegend=False)
    return _fixed(fig, height=320)
