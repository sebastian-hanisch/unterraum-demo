"""Unterraumverfahren: MUSIC und ESPRIT trennen Linien jenseits des Periodogramms - wenn die Annahmen stimmen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 6 der Reihe „Spektral- und Wavelet-Analyse“. Ein simuliertes Signal (bis zu drei reine Linien in weißem oder gefärbtem Rauschen) wird mit MUSIC, Wurzel-MUSIC und ESPRIT
ausgewertet; gezeigt werden Eigenwerte, Linienzahl nach AIC und MDL, Genauigkeit gegen die Cramér-Rao-Schranke und das Scheitern bei falschen Annahmen. Aufbau und Regeln:
DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import ure_constants as C
import ure_evaluation as ev
from ure_evaluation import Settings, analyse, de
from ure_pdf_export import build_pdf
from ure_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from ure_visualization import (
    build_accuracy,
    build_coloured,
    build_count,
    build_eigenvalues,
    build_errors,
    build_height,
    build_matrix,
    build_order,
    build_resolution,
    build_spectrum,
)

st.set_page_config(page_title="Unterraumverfahren – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Rechne die Experimente …")
def _experiment(name):
    return getattr(ev, "sweep_" + name)()


st.title("🔭 Unterraumverfahren: Linien jenseits des Periodogramms")
st.markdown(
    """
Das Periodogramm und seine Verwandten fragen nach der Leistung bei jeder Frequenz. Wer weiß, dass das Signal aus **wenigen reinen Linien in weißem Rauschen** besteht, kann mehr herausholen:
Die **Kovarianzmatrix** der Daten zerfällt in einen **Signalunterraum** (so viele Dimensionen wie Komponenten, zwei je reelle Linie) und einen **Rauschunterraum**. **MUSIC** sucht die Frequenzen,
deren Schwingung auf dem Rauschunterraum senkrecht steht; **Wurzel-MUSIC** und **ESPRIT** lesen sie als Nullstellen bzw. Eigenwerte ab. Zwei Linien im Abstand **eines** Bins, die jedes
Periodogramm verschmiert, sind dann trennbar – **solange das Modell stimmt**: weißes Rauschen, bekannte Linienzahl, genug Daten.

Dies ist die sechste Demo der Reihe „Spektral- und Wavelet-Analyse“, auf die **AR-Spektren** und die Demos davor aufbauend. Alles ist **simuliert**: Lage, Zahl und Stärke der Linien sind bekannt.
Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktionieren die Unterraumverfahren", expanded=True):
    st.markdown(
        """
1. **Schnappschüsse:** Aus den Daten entstehen überlappende Abschnitte der Länge *m*; ihre Kovarianzmatrix (mit Vorwärts-Rückwärts-Mittel) wird zerlegt.
2. **Unterräume:** Die größten Eigenwerte gehören zum Signal (zwei je reelle Linie), alle übrigen liegen bei der Rauschvarianz. Die **Linienzahl** schätzen **AIC** und **MDL** aus diesen Eigenwerten.
3. **Drei Wege zu den Frequenzen:** **MUSIC** (Pseudospektrum mit Gipfeln), **Wurzel-MUSIC** (Polynom-Nullstellen), **ESPRIT** (Eigenwerte einer Verschiebungsmatrix).
4. **Preis:** Die Verfahren liefern Frequenzen, kein Leistungsspektrum – die Höhe eines Gipfels ist keine Leistung. Falsche Ordnung oder gefärbtes Rauschen verschieben und vermehren die Gipfel.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Signal**")
    n = st.select_slider("Länge n der Aufzeichnung", options=list(C.N_CHOICES), key="n_select", help="Anzahl der Messwerte (Abtastrate 1000 Hz).")
    lines = st.radio("Zahl der Linien", list(C.LINE_COUNTS), key="lines_select", horizontal=True, help="Gleich starke reine Töne; die Linie 3 liegt 0,13 f_s von Linie 1 entfernt.")
    line = st.slider("Linie 1 (Bruchteil der Abtastrate)", C.LINE_MIN, C.LINE_MAX, key="line_slider", step=C.LINE_STEP, format="%.4f", help="Lage der ersten Linie; mal 1000 Hz ergibt die Frequenz.")
    if lines >= 2:
        gap = st.slider("Abstand der Linie 2 (Bins von f_s/n)", C.GAP_MIN, C.GAP_MAX, float(st.session_state["gap_kept"]), step=C.GAP_STEP,
                        help="Abstand der zweiten Linie in Vielfachen von f_s/n. Ein Periodogramm mit Hann-Fenster trennt sie ab etwa 3 Bins.")
        st.session_state["gap_kept"] = float(gap)
    else:
        gap = float(st.session_state["gap_kept"])
    snr = st.slider("Jede Linie gegen Rauschen (dB je Abtastwert)", C.SNR_MIN, C.SNR_MAX, key="snr_slider", help="Linienleistung gegen die Rauschvarianz 1. 10 dB: deutlich über dem Rauschen.")
    background = st.selectbox("Hintergrund", list(C.BACKGROUNDS), key="background_select",
                              help="Weißes Rauschen ist die Annahme der Verfahren. Gefärbt: Resonanz bei 200 Hz, mit gleicher oder zehnfacher Leistung gegen das weiße Niveau.")
    st.markdown("**Verfahren**")
    m = st.selectbox("Größe m der Kovarianzmatrix", list(C.M_CHOICES), key="m_select", help=f"Bruchteil von n, mindestens {C.M_FLOOR}, höchstens n/2. Größer: schärfer, aber weniger Schnappschüsse.")
    order_mode = st.selectbox("Zahl der Komponenten", list(C.ORDER_MODES), key="order_mode_select",
                              help="„fest“: Sie wählen p (zwei je Linie). AIC oder MDL: p folgt aus den Eigenwerten (bei ungerader Zahl wird aufgerundet).")
    if order_mode == "fest":
        order = st.select_slider("Komponenten p (zwei je Linie)", options=list(C.ORDER_CHOICES), value=int(st.session_state["order_kept"]), help="Dimension des Signalunterraums.")
        st.session_state["order_kept"] = int(order)
    else:
        order = int(st.session_state["order_kept"])
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_select": int(n), "lines_select": int(lines), "line_slider": float(line), "gap_kept": float(gap), "snr_slider": int(snr), "background_select": background,
                   "m_select": m, "order_mode_select": order_mode, "order_kept": int(order), "seed_input": int(seed)})

settings = Settings(int(n), int(lines), float(line), float(gap), int(snr), background, m, order_mode, int(order), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Komponenten im Modell", f"{a.p}", delta=("fest gewählt" if order_mode == "fest" else f"{order_mode} wählt {a.p}"), delta_color="off", help=f"Wahr: {2 * settings.lines}.")
m2.metric("Linienzahl nach AIC / MDL", f"{ev.half(a.k_aic)} / {ev.half(a.k_mdl)}", delta=f"wahr {settings.lines}", delta_color="off", help="Komponenten geteilt durch zwei.")
e_esprit = a.err_bins["ESPRIT"]
if e_esprit is not None:
    m3.metric("Fehler ESPRIT (größter)", f"{de(max(abs(x) for x in e_esprit), 3)} Bins", delta=(f"Schranke {de(a.crb, 3)} Bins" if a.crb is not None else "Cramér-Rao-Schranke nur bei einer Linie"),
              delta_color="off", help="Bins von 1/n.")
else:
    m3.metric("Fehler ESPRIT (größter)", "–", delta="Zahl der Frequenzen stimmt nicht", delta_color="off")
m4.metric("Rechenaufwand", f"{a.ops:,}".replace(",", "."), delta=f"{de(a.ops / a.ops_fft, 1)}-fach einer FFT der Länge n", delta_color="off", help="Multiplikationen für Kovarianz und Eigenzerlegung (grobe Zählung).")
st.plotly_chart(build_spectrum(a, zoom=False), width="stretch", key="spectrum_full")
st.caption("Blau: MUSIC-Pseudospektrum, grau: Periodogramm (Hann), orange: AR-Spektrum (Burg, Ordnung 16); jedes gegen das eigene Maximum. Rot gestrichelt: wahre Lage der Linien, "
           "Rauten: ESPRIT-Frequenzen. Die Höhe eines Pseudospektrums ist keine Leistung.")
st.plotly_chart(build_spectrum(a, zoom=True), width="stretch", key="spectrum_zoom")
st.caption("Ausschnitt um die Linien.")
pc1, pc2 = st.columns(2)
pc1.plotly_chart(build_eigenvalues(a), width="stretch", key="eigenvalues")
pc2.plotly_chart(build_errors(a), width="stretch", key="errors")
st.caption("Links: Eigenwerte der Kovarianzmatrix (violett: Signalunterraum, grau: Rauschunterraum). Rechts: Frequenzfehler je wahrer Linie in Bins von 1/n"
           + (f"; gestrichelt die Cramér-Rao-Schranke ±{de(a.crb, 3)} Bins." if a.crb is not None else "."))
st.dataframe(pd.DataFrame([{"Verfahren": k, "Frequenzen (Hz)": ", ".join(de(float(f), 2) for f in v), "gefunden": "ja" if a.found[k] else "nein"} for k, v in a.est_hz.items()]
                          + [{"Verfahren": "wahr", "Frequenzen (Hz)": ", ".join(de(f, 2) for f in a.true_hz), "gefunden": ""}]), hide_index=True, width="stretch")
st.caption("„Gefunden“ heißt: die Zahl der Frequenzen stimmt, und jede liegt innerhalb eines Viertels des Linienabstands (bei einer Linie: eines halben Bins) neben der Wahrheit. "
           "Das Periodogramm bekommt dieselbe Linienzahl zugewiesen wie die Unterraumverfahren.")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="unterraum.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was gewinnen die Unterraumverfahren, und was setzen sie voraus?")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption(f"Feste Zufallsströme (Seed {C.SWEEP_SEED} und folgende), unabhängig von den Reglern. Abtastrate {C.FS:g} Hz; Linien mit Amplitude 1 und zufälliger Phase, weißes Rauschen. "
           "Frequenzfehler in Bins von 1/n.")

st.markdown("#### Auflösung: zwei Linien dicht beieinander")
res = _experiment("resolution")
st.caption(f"Zwei gleich starke Linien ({C.EXP_RES_SNR:g} dB je Linie), n = {C.EXP_RES_N}, {C.EXP_RES_RUNS} Läufe, bekannte Linienzahl. „Gefunden“: beide Frequenzen innerhalb von Δ/4 (Δ: Abstand). "
           "„Aufgelöst“: zwei Maxima mit Einbruch von mindestens 3 dB im Spektrum. Matrixgröße m = n/3 bzw. n/2.")
st.dataframe(pd.DataFrame({"Abstand (Bins)": [de(d, 2) for d in res[0]["deltas"]], **{r["label"]: [f"{de(100 * v, 0)} %" for v in r["shares"]] for r in res}}), hide_index=True, width="stretch")
st.plotly_chart(build_resolution(res), width="stretch", key="exp_resolution")
kind, text = ev.verdict_resolution(res)
getattr(st, kind)(text)

st.markdown("#### Genauigkeit einer einzelnen Linie gegen die Cramér-Rao-Schranke")
acc = _experiment("accuracy")
st.caption(f"Eine Linie, Matrix n/3, {C.EXP_ACC_RUNS} Läufe je Zelle. Gezeigt: RMSE der Frequenz geteilt durch die Schranke (Fehler ab {C.EXP_ACC_OUTLIER:g} Bins als Ausreißer ausgenommen, Anteil in Klammern). "
           "Schwellenverhalten: Bei n = 64 steigt der Fehler unterhalb von etwa 0 dB weit über die Schranke, bei −6 dB kommen 10 bis 20 % Ausreißer hinzu.")
st.dataframe(pd.DataFrame([{"n": r["n"], "SNR (dB)": de(r["snr"], 0), "Schranke (Bins)": de(r["crb"], 4),
                            **{k: f"{de(v['ratio'], 2)} ({de(100 * v['outliers'], 0)} %)" for k, v in r["methods"].items()}} for r in acc]), hide_index=True, width="stretch")
st.plotly_chart(build_accuracy(acc), width="stretch", key="exp_accuracy")
kind, text = ev.verdict_accuracy(acc)
getattr(st, kind)(text)

st.markdown("#### Wie viele Linien? AIC gegen MDL")
cnt = _experiment("count")
st.caption(f"n = {C.EXP_COUNT_N}, Matrix {C.EXP_COUNT_M}, {C.EXP_COUNT_RUNS} Läufe; Anteil der Läufe, in denen die geschätzte Zahl der Komponenten gleich zwei je Linie ist (richtig), darunter (zu klein) oder darüber (zu groß).")
st.dataframe(pd.DataFrame([{"Linien": r["lines"], "SNR (dB)": de(r["snr"], 0),
                            **{f"{k}: {t}": f"{de(100 * r[k][key], 0)} %" for k in ("AIC", "MDL") for t, key in (("richtig", "right"), ("zu klein", "under"), ("zu groß", "over"))}} for r in cnt]),
             hide_index=True, width="stretch")
st.plotly_chart(build_count(cnt), width="stretch", key="exp_count")
kind, text = ev.verdict_count(cnt)
getattr(st, kind)(text)

st.markdown("#### Größe der Kovarianzmatrix")
mat = _experiment("matrix")
st.caption(f"Zwei Linien im Abstand {C.EXP_MATRIX_DELTA:g} Bins, n = {C.EXP_MATRIX_N}, {C.EXP_MATRIX_RUNS} Läufe, Ordnung 4 bekannt. RMSE nur über die Läufe, in denen beide Linien gefunden wurden.")
st.dataframe(pd.DataFrame([{"m": r["m"], "MUSIC: gefunden": f"{de(100 * r['MUSIC']['found'], 0)} %", "MUSIC: RMSE (Bins)": de(r["MUSIC"]["rmse"], 3), "ESPRIT: gefunden": f"{de(100 * r['ESPRIT']['found'], 0)} %",
                            "ESPRIT: RMSE (Bins)": de(r["ESPRIT"]["rmse"], 3)} for r in mat]), hide_index=True, width="stretch")
st.plotly_chart(build_matrix(mat), width="stretch", key="exp_matrix")
kind, text = ev.verdict_matrix(mat)
getattr(st, kind)(text)

st.markdown("#### Wenn das Rauschen nicht weiß ist")
col = _experiment("coloured")
st.caption(f"Zwei weit getrennte Linien (bei {C.EXP_COLOUR_FREQS[0] * C.FS:g} und {C.EXP_COLOUR_FREQS[1] * C.FS:g} Hz), n = {C.EXP_COLOUR_N}, Matrix {C.EXP_COLOUR_M}, {C.EXP_COLOUR_RUNS} Läufe; "
           "gefärbter Hintergrund: Resonanz bei 200 Hz. „Gefunden“: beide Frequenzen innerhalb von 10 Hz (bei AR: unter den drei höchsten Gipfeln).")
st.dataframe(pd.DataFrame([{"Hintergrund": r["label"], "MUSIC": f"{de(100 * r['MUSIC'], 0)} %", "ESPRIT": f"{de(100 * r['ESPRIT'], 0)} %", "AR (Burg, p = 16)": f"{de(100 * r['Burg'], 0)} %"} for r in col]),
             hide_index=True, width="stretch")
st.plotly_chart(build_coloured(col), width="stretch", key="exp_coloured")
kind, text = ev.verdict_coloured(col)
getattr(st, kind)(text)

st.markdown("#### Wenn die Ordnung nicht stimmt")
orr = _experiment("order")
st.caption(f"Zwei Linien (wahr: vier Komponenten), n = {C.EXP_ORDER_N}, Matrix {C.EXP_ORDER_M}, {C.EXP_ORDER_RUNS} Läufe; MUSIC mit gewählter Ordnung p. Zusatzgipfel: ein weiterer Gipfel über 10 % der höchsten Höhe.")
st.dataframe(pd.DataFrame([{"p": r["p"], "Frequenzfehler RMSE (Bins)": de(r["rmse"], 3), "Ausreißer": f"{de(100 * r['outliers'], 1)} %", "Zusatzgipfel": f"{de(100 * r['extra'], 0)} %"} for r in orr]),
             hide_index=True, width="stretch")
st.plotly_chart(build_order(orr), width="stretch", key="exp_order")
kind, text = ev.verdict_order(orr)
getattr(st, kind)(text)

st.markdown("#### Die Höhe ist keine Leistung")
hgt = _experiment("height")
st.caption(f"Zwei Linien, die schwache 20 dB unter der starken, n = {C.EXP_HEIGHT_N}, Matrix {C.EXP_HEIGHT_M}, {C.EXP_HEIGHT_RUNS} Läufe; Höhenunterschied der Gipfel (Mittel, Streuung als Balken).")
st.plotly_chart(build_height(hgt), width="stretch", key="exp_height")
kind, text = ev.verdict_height(hgt)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Rauschen ist weiß** | Ein gefärbter Hintergrund hebt Eigenwerte an: Gipfel verschieben sich oder fehlen, die Linienzahl wird überschätzt. | AR-Spektren, Prewhitening |
| **Die Linienzahl ist bekannt** | Zu kleine Ordnung: Linien fehlen, Frequenzen sind falsch. AIC überschätzt, MDL unterschätzt bei schwachen Linien. | Konfidenz und Falschalarm |
| **Die Linien sind gleichzeitig vorhanden** | Kohärente (korrelierte) Quellen verringern den Rang des Signalunterraums; das Vorwärts-Rückwärts-Mittel hilft nur teilweise. | – |
| **Genug Daten und Rauschabstand** | Unterhalb einer Schwelle (etwa 0 dB je Abtastwert bei n = 64) steigt der Fehler weit über die Cramér-Rao-Schranke und Ausreißer treten auf; die Schranke gilt nur darüber. | Schätztheorie |
| **Eine Linie ist kein Spektrum** | Das Pseudospektrum zeigt Lagen, nicht Leistungen; Höhen sind nicht vergleichbar. | Periodogramm, Multitaper |
| **Stationär** | Wandern die Frequenzen, mittelt die Kovarianzmatrix über die ganze Aufzeichnung. | Spektrogramm |
""",
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $x[t] = \sum_{j=1}^{q} A_j\cos(2\pi f_j t + \varphi_j) + e[t]$, $e$ weißes Rauschen mit Varianz $\sigma^2$. Eine reelle Linie belegt die Komponenten $\pm f_j$; $p = 2q$.

**Kovarianzmatrix.** Mit den Schnappschüssen $\mathbf{x}_t = (x[t],\dots,x[t+m-1])^\top$ ist $\hat R = \frac1N\sum_t \mathbf{x}_t\mathbf{x}_t^\top$, $N = n - m + 1$; mit der Spiegelung $J$ dient
$(\hat R + J\hat R J)/2$ der Vorwärts-Rückwärts-Mittelung. Eigenzerlegung $\hat R = \sum_i \lambda_i \mathbf{v}_i\mathbf{v}_i^\top$, $\lambda_1\ge\dots\ge\lambda_m$; Signalunterraum $E_s = (\mathbf v_1..\mathbf v_p)$, Rauschunterraum $E_n$.

**MUSIC.** $P(f) = 1/\lVert E_n^\top \mathbf a(f)\rVert^2$ mit $\mathbf a(f) = (1, e^{-2\pi i f},\dots,e^{-2\pi i f(m-1)})^\top$; die Gipfel liegen bei den Linien.

**Wurzel-MUSIC.** Nullstellen des Polynoms $\sum_k c_k z^k$ mit den Diagonalsummen $c_k$ von $E_nE_n^\top$; die $p$ Wurzeln innerhalb des Einheitskreises, die ihm am nächsten liegen, haben die Winkel $2\pi f_j$.

**ESPRIT.** Mit $U_1$, $U_2$ (Signalvektoren ohne letzte bzw. erste Zeile) löst $U_1\Phi\approx U_2$ im Sinne kleinster Quadrate; die Eigenwerte von $\Phi$ sind $e^{\pm 2\pi i f_j}$.

**Linienzahl (Wax und Kailath).** Mit $g$ und $a$ als geometrischem und arithmetischem Mittel der $m-k$ kleinsten Eigenwerte: $\mathrm{MDL}(k) = -N(m-k)\ln(g/a) + \tfrac12 k(2m-k)\ln N$,
$\mathrm{AIC}(k) = -2N(m-k)\ln(g/a) + 2k(2m-k)$; $\hat p$ minimiert das Kriterium (hier mit $N = n-m+1$ als Näherung für unabhängige Schnappschüsse).

**Cramér-Rao-Schranke** (eine reelle Linie, SNR $= A^2/(2\sigma^2)$): $\mathrm{Std}(\hat f) \ge \sqrt{\dfrac{3}{\pi^2\,\mathrm{SNR}\,n(n^2-1)}}$ (in Bins mit Faktor $n$).

**Quellen.** Lehrbuchwissen (Schmidt 1986; Roy und Kailath 1989; Wax und Kailath 1985; Rife und Boorstyn 1974; Stoica und Moses 2005). Formeln hier per Simulation, Handrechnung und gegen scipy geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
