"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from ure_evaluation import Analysis, de

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->", "σ": "sigma", "Σ": "Summe", "±": "+-", "√": "Wurzel aus ", "×": "x", "é": "e"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def _errors(err) -> str:
    return "Zahl der Frequenzen falsch" if err is None else ", ".join(de(e, 3) for e in err)


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Unterraumverfahren: Einstellung und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    freqs = ", ".join(de(f, 1) for f in a.true_hz)
    order = f"fest {a.p}" if s.order_mode == "fest" else f"{s.order_mode} wählt {a.p}"
    pdf.multi_cell(0, 5, _s(f"Folge der Länge n = {s.n}, Linien bei {freqs} Hz ({s.snr} dB je Abtastwert), Hintergrund: {s.background}, Matrix {a.m} x {a.m}, Ordnung: {order}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    out = [f"Zahl der Komponenten nach AIC: {a.k_aic}, nach MDL: {a.k_mdl} (wahr: {2 * s.lines}).",
           *[f"Fehler der Frequenzen in Bins ({k}): {_errors(v)}; gefunden: {'ja' if a.found[k] else 'nein'}." for k, v in a.err_bins.items()]]
    if a.crb is not None:
        out.append(f"Cramér-Rao-Schranke der Frequenz einer Linie: {de(a.crb, 4)} Bins.")
    if a.resolved is not None:
        out.append("Zwei Linien getrennt: " + ", ".join(f"{k} {'ja' if v else 'nein'}" for k, v in a.resolved.items()) + ".")
    out.append(f"Rechenaufwand (Multiplikationen): {a.ops} (Kovarianz und Eigenzerlegung), {a.ops_fft} (eine FFT der Länge n).")
    for line in out:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: unterraum-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
