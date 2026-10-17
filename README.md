# Unterraumverfahren: MUSIC und ESPRIT trennen Linien jenseits des Periodogramms – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-unterraum-demo.streamlit.app/)** -->

Sechste Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, aufbauend auf den Demos
zum **Periodogramm**, zu **Fenstern**, zu **Bartlett und Welch**, zum **Multitaper** und zum **AR-Spektrum**. Ein simuliertes Signal (null bis drei reine Linien in weißem oder gefärbtem Rauschen)
wird mit **MUSIC**, **Wurzel-MUSIC** und **ESPRIT** ausgewertet; Eigenwerte, Linienzahl nach AIC und MDL, Genauigkeit gegen die Cramér-Rao-Schranke und das Scheitern bei falschen Annahmen
werden gegen die Wahrheit gemessen.

**Namensfalle:** „Unterraum“ meint hier den Signal- und Rauschunterraum der Kovarianzmatrix, nicht Unterräume in der Optimierung.

## Warum dieses Problem

Das Periodogramm trennt zwei Linien erst ab etwa zwei bis drei Bins Abstand. Wer weiß, dass das Signal aus wenigen reinen Linien in weißem Rauschen besteht, kann mehr herausholen. Die Demo zeigt,
wie viel (Linien im Abstand eines Bins), und woran der Gewinn hängt (weißes Rauschen, bekannte Linienzahl, genug Daten und Rauschabstand).

## Modell

Null bis drei gleich starke Sinuslinien (Lage frei, Abstand der zweiten in Bins von f_s/n, die dritte 0,13 f_s von der ersten), zufällige Phasen, Rauschvarianz 1; Hintergrund weiß oder gefärbt
(Resonanz bei 200 Hz, gleiche oder zehnfache Leistung gegen das weiße Niveau). Kovarianzmatrix aus überlappenden Schnappschüssen mit Vorwärts-Rückwärts-Mittel, Größe m = n/6 bis n/2 (mindestens 12).
MUSIC mit grober und feiner Gittersuche, Wurzel-MUSIC, ESPRIT (Kleinste Quadrate), Linienzahl nach Wax und Kailath – alles selbst in numpy; scipy dient nur als Orakel in den Tests.
Die Abtastrate ist fest 1000 Hz.

## Methodik

n = 32 bis 512. Vergleich im Bild mit dem Hann-Periodogramm und dem AR-Spektrum (Burg, Ordnung 16), in den Experimenten zusätzlich mit dem Multitaper (NW = 2). Experimente über feste Zufallsströme
(Seeds ab 600000), jede Zahl ein Mittel über 100 bis 200 Läufe (Standardfehler eines Anteils höchstens 5 Prozentpunkte). „Gefunden“: Zahl der Frequenzen stimmt, jede liegt innerhalb eines Viertels
des Linienabstands neben der Wahrheit; „aufgelöst“: zwei Maxima mit einem Einbruch von mindestens 3 dB.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und `tests/test_experiments_golden.py`.

| Frage | Befund |
|---|---|
| Wie dicht dürfen zwei Linien liegen (10 dB je Linie, n = 64, Linienzahl bekannt)? | Bei 1 Bin Abstand finden MUSIC und Wurzel-MUSIC beide Linien in 99,5 % der Läufe, ESPRIT in 100 %; das Hann-Periodogramm trennt sie in 20 %, Burg (Ordnung 16) in 56,5 %, der Multitaper (NW = 2) nie. Bei 0,5 Bins: ESPRIT 21 % (Matrix n/3), 52,5 % (Matrix n/2). |
| Wie genau ist die Frequenz einer einzelnen Linie (n = 256, 5 dB)? | Das 0,92-fache der Cramér-Rao-Schranke (Rechteck-Periodogramm), 1,09-fache (MUSIC), 0,95-fache (ESPRIT): Für eine einzelne Linie gewinnen die Unterraumverfahren nichts. Bei n = 64 und 0 dB: 1,13 (Rechteck-Periodogramm), 1,48 (ESPRIT), 1,57 (MUSIC), 1,69 (Hann-Periodogramm). |
| Wo bricht die Genauigkeit ein (n = 64, −6 dB je Abtastwert)? | Das Rechteck-Periodogramm liegt beim 1,8-fachen der Schranke bei 10 % Ausreißern (Fehler ab 3 Bins), ESPRIT beim 3,4-fachen bei 20 %. |
| Wie viele Linien? (zwei Linien, 0 dB, n = 128, Matrix 32) | MDL trifft die Zahl der Komponenten in 98 % der Läufe, AIC in 67 % (zu groß in 33 %). Bei −9 dB: MDL 5 %, AIC 25 %; MDL unterschätzt (95 %). |
| Wie groß muss die Matrix sein (zwei Linien, 2 Bins, n = 64)? | m = 6: MUSIC 48 %, ESPRIT 49,5 % gefunden; m = 8: 89,5 / 92,5 %; ab m = 12 100 %. ESPRIT-Fehler 0,067 Bins bei m = 12, 0,025 Bins bei m = 24. |
| Und bei gefärbtem Rauschen (zehnfache Leistung, zwei weit getrennte Linien, n = 128)? | MUSIC findet beide Linien in 75,5 %, ESPRIT in 67 %, AR (Burg, Ordnung 16) in 100 %; bei gleicher Leistung wie das weiße Rauschen alle drei 100 %. |
| Und bei falscher Ordnung (zwei Linien, n = 128)? | Ordnung 2 statt 4: Frequenzfehler 0,116 Bins, in 64,5 % der Läufe ein Zusatzgipfel; Ordnung 3: 17,5 %; Ordnung 4: 0,044 Bins ohne Zusatzgipfel; Ordnung 12: 0,045 Bins, 2 % Zusatzgipfel. |
| Ist die Gipfelhöhe eine Leistung? | Eine Linie 20 dB unter der starken liegt im Periodogramm bei −20,0 dB, im MUSIC-Pseudospektrum bei −11,9 dB (Streuung 4,2 dB). |

## Ehrliche Grenzen

- **Heimvorteil:** Die Daten sind exakt Linien plus weißes Rauschen. Gefärbtes Rauschen und falsche Ordnung sind das Gegengewicht, aber nur für zwei Arten der Verletzung (keine kohärenten Linien, keine
  sich ändernden Frequenzen).
- **ESPRIT in der Kleinste-Quadrate-Fassung, MUSIC mit Vorwärts-Rückwärts-Mittel:** Andere Fassungen (TLS, Unterraum-Glättung) verschieben die Zahlen.
- **Die Schnappschüsse überlappen und sind nicht unabhängig;** AIC und MDL sind Näherungen mit N = n − m + 1.
- **Reelle Linien:** doppelte Eigenwerte je Linie (±f); Linien nahe 0 und der Nyquist-Frequenz liegen nicht in den Experimenten.
- **Die Cramér-Rao-Schranke** gilt asymptotisch und fern von 0 und Nyquist; die gemessenen Verhältnisse unter 1 (0,92 bis 0,95) liegen im Rahmen des Stichprobenfehlers von 100 Läufen (etwa ±7 %).
- **Die Auflösung hängt vom Einbruchskriterium, von 10 dB je Linie und von n = 64 ab.**
- **Quellen** (Schmidt 1986, Roy und Kailath 1989, Wax und Kailath 1985, Rife und Boorstyn 1974): nicht gelesen; Formeln per Simulation, Handrechnung und gegen scipy geprüft.

## Verifikation

- **Kern gegen Handrechnung:** x = (1, 0, −1, 0, 1, 0), m = 3: vier Schnappschüsse, R = [[0,5; 0; −0,5], [0; 0,5; 0], [−0,5; 0; 0,5]], Eigenwerte 1 / 0,5 / 0, Pol des Pseudospektrums bei f = 1/4, ESPRIT und Wurzel-MUSIC liefern 1/4.
- **Orakel (`tests/test_oracle_subspace.py`, `tests/test_compare.py`):** Kovarianzmatrix gegen die Doppelsumme, Eigenzerlegung gegen `scipy.linalg.eigh`, rauschfreie Linien auf 1e-8 (ESPRIT, Wurzel-MUSIC) bzw.
  2e-5 (MUSIC) wiedergefunden mit genau 2q von null verschiedenen Eigenwerten, Linienzahl gegen eine unabhängige Schleife über die Eigenwerte, Taper gegen `scipy.signal.windows.dpss`, Burg gegen eine
  unabhängige stufenweise Minimierung.
- **Generator** (`tests/test_generator.py`): wortgleich aus der ersten Demo.
- **Zahlen:** jede Zahl dieser README in `tests/test_claims.py`; Festwerte der Experimente in `tests/test_experiments_golden.py`; Verdikte in `tests/test_verdicts.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen), Auswahlfelder, Permalink, bedingte Regler, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `signal_generator.py` | Gemeinsamer Generator der Reihe (wortgleich) |
| `ure_subspace.py` | Schnappschüsse, Kovarianz, Eigenzerlegung, MUSIC, Wurzel-MUSIC, ESPRIT, Linienzahl, Cramér-Rao-Schranke |
| `ure_compare.py` | Fenster, Slepian-Taper, Multitaper, Burg, AR-Spektrum, Auflösungskriterium (wortgleich aus den Vorgänger-Demos) |
| `ure_constants.py` | Regler-Grenzen, feste Annahmen, Experimente, Presets |
| `ure_evaluation.py` | Analyse, Experimente, Verdikte |
| `ure_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `ure_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `ure_pdf_export.py` | PDF der aktuellen Einstellung (erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `tests/` | Kern, Orakel, Generator, Zahlen, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Kohärente Linien, Unterraum-Glättung, TLS-ESPRIT, Min-Norm, Capon:** nicht enthalten.
- **Konfidenzintervalle und Falschalarm der Linienzahl:** Thema einer Folgedemo.
- **Gemessene Daten:** alles ist simuliert, damit die Wahrheit bekannt ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer. Die Experimente brauchen beim ersten Aufruf rund 60 Sekunden (danach zwischengespeichert).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
