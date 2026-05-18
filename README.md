# NBA Mock Draft Predictor — README

Predice la **overall pick** di un giocatore fittizio nel Draft NBA usando un modello **Random Forest**.

---

## File del progetto

```
nba_draft.py          ← codice sorgente principale
nba_model.pkl         ← modello già allenato (si carica all'avvio, non si riallena)
nbaplayersdraft.csv   ← dataset di training (1922 giocatori, 1989-2021)
README.md             ← questo file
```

---

## Installazione e avvio

```bash
pip install pandas scikit-learn matplotlib
python nba_draft.py
```

Python 3.8+ richiesto. Tkinter è incluso di default.

---

## Workflow generale

```
nbaplayersdraft.csv
        │
        ▼
  carica_o_allena_modello()
        │
        ├─ se nba_model.pkl esiste → lo carica (< 1 secondo)
        └─ se NON esiste            → allena RF e lo salva
                │
                ▼
        RandomForestRegressor.fit(X, y)
                │
                ▼
        nba_model.pkl  (salvato su disco)
                │
                ▼
        UI Tkinter
                │
        utente inserisce statistiche
                │
                ▼
        calcola_score()       ← percentile pesato su ogni statistica
                │
                ▼
        score_in_pick()       ← curva non lineare score → pick 1-60
                │
                ▼
        mostra_risultati() + mostra_grafico()
```

---

## Spiegazione del codice riga per riga

### Importazioni

```python
import tkinter as tk                         # libreria per la finestra grafica
from tkinter import messagebox               # finestre di errore/avviso
import pandas as pd                          # lettura e manipolazione del CSV
import numpy as np                           # calcoli numerici
import matplotlib.pyplot as plt              # grafici
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # integra matplotlib in tkinter
import matplotlib.patches as mpatches       # patch colorate per la legenda
from sklearn.ensemble import RandomForestRegressor  # modello AI
import pickle                               # salva/carica oggetti Python su disco
import os                                   # controlla se un file esiste
```

### Costanti

```python
MODEL_FILE = "nba_model.pkl"    # nome del file dove salviamo il modello
CSV_FILE   = "nbaplayersdraft.csv"

FEATURES = [...]   # lista delle 10 statistiche usate dal modello

PESI = {           # quanto conta ogni statistica nello score finale (somma = 1.0)
    "points_per_game": 0.25,           # PPG pesa il 25%
    "win_shares_per_48_minutes": 0.20, # efficienza pesa il 20%
    "box_plus_minus": 0.18,            # impatto sul punteggio pesa il 18%
    ...
}

CAMPI = [...]   # definisce i campi della UI: nome, etichetta, min, max, default

ZONE = [        # zone del draft con soglie e colori
    (5,  "Top-5",   "#FFD700"),   # pick 1-5:  oro
    (14, "Lottery", "#4C72B0"),   # pick 6-14: blu
    ...
]
```

### `carica_o_allena_modello()`

```python
def carica_o_allena_modello():
    if os.path.exists(MODEL_FILE):       # controlla se il file .pkl esiste già
        with open(MODEL_FILE, "rb") as f:
            dati = pickle.load(f)        # carica il modello dal disco (< 1 secondo)
        return dati["model"], dati["df"]
    else:
        return allena_e_salva()          # prima volta: allena e salva
```

### `allena_e_salva()`

```python
def allena_e_salva():
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=FEATURES + ["overall_pick"])  # scarta righe incomplete

    X = df[FEATURES].values        # input:  matrice 1922 × 10
    y = df["overall_pick"].values  # output: vettore con la pick di ogni giocatore

    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    # n_estimators=100 → usa 100 alberi decisionali
    # max_depth=8      → ogni albero è profondo al massimo 8 livelli
    # random_state=42  → seed fisso per risultati riproducibili

    model.fit(X, y)    # ALLENAMENTO: il modello impara la relazione stats → pick

    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "df": df}, f)  # salva tutto su disco
```

### `calcola_score()`

```python
def calcola_score(statistiche, df_riferimento):
    score = 0.0
    for feature, peso in PESI.items():
        valore     = statistiche[feature]                     # valore inserito dall'utente
        colonna    = df_riferimento[feature].dropna().values  # tutti i valori nel dataset
        percentile = (colonna < valore).mean()
        # percentile = frazione di giocatori nel dataset con quella statistica inferiore
        # es: PPG=27 → il 98% dei giocatori ha meno punti → percentile = 0.98
        score += percentile * peso    # contributo pesato alla valutazione finale
    return score   # 0.0 = peggiore assoluto, 1.0 = migliore assoluto
```

**Esempio pratico:**
| Stat | Valore | Percentile | Peso | Contributo |
|---|---|---|---|---|
| PPG | 27 | 0.98 | 0.25 | 0.245 |
| WS/48 | 0.23 | 0.95 | 0.20 | 0.190 |
| BPM | 9 | 0.97 | 0.18 | 0.175 |
| … | … | … | … | … |
| **Totale** | | | | **≈ 0.98 → pick #2** |

### `score_in_pick()`

```python
def score_in_pick(score):
    pick = np.interp(
        score,
        [0.0, 0.25, 0.45, 0.60, 0.75, 0.88, 1.0],  # valori di score
        [60,  50,   35,   25,   15,    5,    1  ],   # pick corrispondenti
    )
    return pick
```

`np.interp` fa interpolazione lineare tra i punti della curva:
- score 0.00 → pick 60 (fondo del draft)
- score 0.50 → pick ~32 (secondo turno)
- score 1.00 → pick 1  (prima assoluta)

### Classe `App` — UI Tkinter

```python
class App(tk.Tk):       # eredita da tk.Tk → è la finestra principale

    def __init__(self):
        super().__init__()          # inizializza la finestra Tkinter
        self.title("NBA Mock Draft")
        self.geometry("1100x760")   # dimensioni in pixel

        tk.Label(..., text="Caricamento modello...").pack()
        self.update()               # forza il ridisegno: senza questo la finestra è bianca

        self.model, self.df = carica_o_allena_modello()

        for widget in self.winfo_children():
            widget.destroy()        # rimuove il messaggio di caricamento
        self.costruisci_ui()
```

### `costruisci_ui()`

Costruisce la UI in due colonne usando il layout `grid`:

```python
corpo.columnconfigure(1, weight=1)  # la colonna destra si allarga con la finestra
corpo.rowconfigure(0, weight=1)     # la riga si allunga verticalmente
```

I campi statistiche vengono creati in ciclo:
```python
for feature, etichetta, minimo, massimo, default in CAMPI:
    var = tk.DoubleVar(value=default)               # variabile collegata al campo
    self.input_vars[feature] = (var, minimo, massimo)  # salvata per la lettura
    tk.Entry(riga, textvariable=var, ...).pack(...)    # crea il campo visivo
```

La griglia degli anni usa `tk.BooleanVar` collegato a ogni `Checkbutton`:
```python
for i, anno in enumerate(range(1989, 2022)):
    var = tk.BooleanVar(value=(anno == 2003))  # 2003 selezionato di default
    self.anni_vars[anno] = var                 # salvata per la lettura
    tk.Checkbutton(..., variable=var).grid(row=i // 8, column=i % 8)
    # i // 8 → numero di riga (8 anni per riga)
    # i % 8  → numero di colonna
```

### `analizza()`

```python
def analizza(self):
    # 1. Legge e valida tutti i valori dai campi
    for feature, (var, minimo, massimo) in self.input_vars.items():
        valore = float(var.get())             # legge il valore dalla UI
        if not (minimo <= valore <= massimo): # controlla il range
            messagebox.showerror(...)
            return

    # 2. Raccoglie gli anni selezionati dalle checkbox
    anni_selezionati = sorted([anno for anno, var in self.anni_vars.items() if var.get()])

    # 3. Calcola score e pick base
    score     = calcola_score(statistiche, self.df)
    pick_base = score_in_pick(score)

    # 4. Aggiunge un piccolo offset per ogni anno (variabilità storica)
    media_globale = self.df["overall_pick"].mean()
    for anno in anni_selezionati:
        df_anno = self.df[self.df["year"] == anno]
        offset  = (df_anno["overall_pick"].mean() - media_globale) * 0.08
        # offset piccolo (×0.08): sposta la pick di ±2 al massimo
        pick = int(np.clip(round(pick_base + offset), 1, 60))
        # np.clip garantisce che la pick rimanga sempre tra 1 e 60
```

### `mostra_grafico()`

Disegna due grafici affiancati con matplotlib:

**Grafico 1 – Pick per anno:**
```python
ax1.invert_yaxis()              # pick 1 in cima, 60 in fondo (come un ranking)
ax1.axhspan(prev, lim, ...)     # fascia colorata di sfondo per ogni zona
ax1.scatter(anno, pick, ...)    # punto per ogni anno selezionato
ax1.annotate(f"#{pick}", ...)   # etichetta testuale sopra ogni punto
```

**Grafico 2 – Feature importance:**
```python
importanze = self.model.feature_importances_
# array prodotto dal Random Forest: indica quanto ogni feature
# ha contribuito alle decisioni degli alberi durante il training
ax2.barh([etichette], [importanze], ...)  # barre orizzontali ordinate
```

---

## Perché il modello usa lo score composito invece della pick diretta?

Il dataset contiene errori storici: Kwame Brown (pick #1, 6.6 PPG) e Nikola Jokić (pick #41, MVP). Se il modello imparasse `statistiche → pick_storica`, replica questi errori umani e dà risultati incoerenti per giocatori fenomenali.

La soluzione: calcolare uno **score composito** basato sui percentili, che misura il valore oggettivo del giocatore rispetto a tutti gli altri nel dataset, indipendentemente da come i team NBA lo hanno effettivamente scelto.

---

## Formato del modello salvato

Il file `nba_model.pkl` è un archivio **pickle** (formato binario nativo di Python):

```python
{
    "model": RandomForestRegressor,  # il modello allenato con sklearn
    "df":    pd.DataFrame,           # il dataset per il calcolo dei percentili
}
```

Pickle è stato scelto perché è incluso nella libreria standard di Python (nessuna installazione aggiuntiva) ed è il formato nativo per i modelli scikit-learn.