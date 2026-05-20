import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

MODEL_FILE = "nba_model.pkl"
CSV_FILE   = "nbaplayersdraft.csv"

FEATURES = [
    "points_per_game",
    "average_total_rebounds",
    "average_assists",
    "win_shares_per_48_minutes",
    "box_plus_minus",
    "value_over_replacement",
    "field_goal_percentage",
    "free_throw_percentage",
    "3_point_percentage",
    "years_active",
]

PESI = {
    "points_per_game":           0.25,
    "win_shares_per_48_minutes": 0.20,
    "box_plus_minus":            0.18,
    "value_over_replacement":    0.15,
    "average_assists":           0.08,
    "average_total_rebounds":    0.08,
    "field_goal_percentage":     0.03,
    "free_throw_percentage":     0.01,
    "3_point_percentage":        0.01,
    "years_active":              0.01,
}

CAMPI = [
    ("points_per_game",           "PPG",   0,    40,  18.5),
    ("average_total_rebounds",    "REB",   0,    25,   7.2),
    ("average_assists",           "AST",   0,    15,   4.1),
    ("win_shares_per_48_minutes", "WS/48", -0.5, 0.5,  0.12),
    ("box_plus_minus",            "BPM",   -15,  20,   2.5),
    ("value_over_replacement",    "VORP",  -30,  100, 15.0),
    ("field_goal_percentage",     "FG%",   0,    1,   0.47),
    ("free_throw_percentage",     "FT%",   0,    1,   0.78),
    ("3_point_percentage",        "3P%",   0,    1,   0.36),
    ("years_active",              "ANNI",  1,    25,   8),
]

ZONE = [
    (5,  "Top-5",     "#FFD700"),
    (14, "Lottery",   "#4C72B0"),
    (30, "1° Giro",   "#2ecc71"),
    (45, "2° Giro A", "#f39c12"),
    (60, "2° Giro B", "#e74c3c"),
]

def carica_o_allena_modello():
    if os.path.exists(MODEL_FILE):
        print("Carico modello salvato...")
        with open(MODEL_FILE, "rb") as f:
            dati = pickle.load(f)
        return dati["model"], dati["df"]
    else:
        print("Modello non trovato, alleno da zero...")
        return allena_e_salva()

def allena_e_salva():

    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=FEATURES + ["overall_pick"])  # rimuove righe con dati mancanti

    X = df[FEATURES].values        # matrice input  (1922 giocatori × 10 feature)
    y = df["overall_pick"].values  # vettore target (pick di ogni giocatore)

    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y)

    # Salva modello + dataset in un unico file binario
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "df": df}, f)

    print(f"Modello salvato in {MODEL_FILE}")
    return model, df

def calcola_score(statistiche, df_riferimento):
    """
    Calcola uno score 0-1 per il giocatore fittizio.
    Per ogni statistica calcola in che percentile si trova rispetto
    a tutti i giocatori del dataset, poi fa la media pesata.
    - Score vicino a 1.0 = giocatore fenomenale (top 1%)
    - Score vicino a 0.0 = giocatore scarso (bottom 1%)
    """
    score = 0.0
    for feature, peso in PESI.items():
        valore     = statistiche[feature]
        colonna    = df_riferimento[feature].dropna().values
        percentile = (colonna < valore).mean()  # frazione di giocatori con stat inferiore
        score     += percentile * peso
    return score

def score_in_pick(score):
    """
    Converte lo score 0-1 in una pick 1-60 con curva non lineare.
    I valori intermedi (0.45-0.75) mappano sulla zona lottery/primo giro.
    """
    pick = np.interp(
        score,
        [0.0, 0.25, 0.45, 0.60, 0.75, 0.88, 1.0],  # score
        [60,  50,   35,   25,   15,    5,    1  ],   # pick corrispondente
    )
    return pick

def colore_pick(pick):
    """Restituisce il colore esadecimale della zona draft in base alla pick."""
    for limite, _, colore in ZONE:
        if pick <= limite:
            return colore
    return "#e74c3c"

# ─────────────────────────────────────────────
# INTERFACCIA GRAFICA
# ─────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("NBA Mock Draft")
        self.configure(bg="#0d0d1a")
        self.geometry("1100x760")

        # Mostra messaggio di caricamento mentre il modello viene caricato
        tk.Label(self, text="Caricamento modello...",
                 bg="#0d0d1a", fg="#f5a623", font=("Courier", 12)).pack(pady=40)
        self.update()  # forza il ridisegno della finestra

        self.model, self.df = carica_o_allena_modello()

        # Pulisce la finestra e costruisce la UI vera
        for widget in self.winfo_children():
            widget.destroy()
        self.costruisci_ui()

    def costruisci_ui(self):
        """Costruisce tutta l'interfaccia grafica."""

        # Titolo in alto
        tk.Label(self, text="🏀 NBA MOCK DRAFT", bg="#0d0d1a", fg="#f5a623",
                 font=("Georgia", 18, "bold")).pack(anchor="w", padx=20, pady=(12, 4))

        # Contenitore principale: colonna sinistra (input) + destra (risultati)
        corpo = tk.Frame(self, bg="#0d0d1a")
        corpo.pack(fill="both", expand=True, padx=20, pady=8)
        corpo.columnconfigure(1, weight=1)
        corpo.rowconfigure(0, weight=1)

        # ── COLONNA SINISTRA: input ──
        sinistra = tk.Frame(corpo, bg="#141428", bd=1, relief="solid")
        sinistra.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        # Campo nome giocatore
        tk.Label(sinistra, text="Nome giocatore", bg="#141428", fg="#8888aa",
                 font=("Courier", 8)).pack(anchor="w", padx=12, pady=(12, 0))
        self.nome_var = tk.StringVar(value="Giocatore X")
        tk.Entry(sinistra, textvariable=self.nome_var, bg="#1e1e3a", fg="white",
                 insertbackground="#f5a623", font=("Courier", 11),
                 relief="flat", width=22).pack(padx=12, ipady=5)

        tk.Label(sinistra, text="─" * 26, bg="#141428", fg="#2a2a4a").pack(pady=6)
        tk.Label(sinistra, text="STATISTICHE", bg="#141428", fg="#8888aa",
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=12)

        # Un campo di input per ogni statistica
        self.input_vars = {}
        for feature, etichetta, minimo, massimo, default in CAMPI:
            riga = tk.Frame(sinistra, bg="#141428")
            riga.pack(fill="x", padx=12, pady=2)
            tk.Label(riga, text=etichetta, width=6, bg="#e74c3c", fg="white",
                     font=("Courier", 8, "bold")).pack(side="left", ipady=3, padx=(0, 6))
            var = tk.DoubleVar(value=default)
            self.input_vars[feature] = (var, minimo, massimo)
            tk.Entry(riga, textvariable=var, bg="#1e1e3a", fg="white",
                     insertbackground="#f5a623", font=("Courier", 10),
                     relief="flat", width=10).pack(side="left", ipady=3)

        tk.Label(sinistra, text="─" * 26, bg="#141428", fg="#2a2a4a").pack(pady=6)
        tk.Label(sinistra, text="ANNO DRAFT", bg="#141428", fg="#8888aa",
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=12)

        # Griglia di checkbox: un anno per cella
        griglia_anni = tk.Frame(sinistra, bg="#141428")
        griglia_anni.pack(padx=12, pady=4)
        self.anni_vars = {}
        for i, anno in enumerate(range(1989, 2022)):
            var = tk.BooleanVar(value=(anno == 2003))  # 2003 selezionato di default
            self.anni_vars[anno] = var
            tk.Checkbutton(griglia_anni, text=str(anno), variable=var,
                           bg="#141428", fg="#8888aa", selectcolor="#1e1e3a",
                           activebackground="#141428", font=("Courier", 7),
                           bd=0, highlightthickness=0).grid(row=i // 8, column=i % 8, sticky="w")

        # Bottone che avvia il calcolo
        tk.Button(sinistra, text="▶  ANALIZZA", command=self.analizza,
                  bg="#f5a623", fg="#0d0d1a", font=("Courier", 12, "bold"),
                  relief="flat", cursor="hand2").pack(fill="x", padx=12, pady=14, ipady=10)

        # ── COLONNA DESTRA: risultati ──
        destra = tk.Frame(corpo, bg="#0d0d1a")
        destra.grid(row=0, column=1, sticky="nsew")
        destra.rowconfigure(1, weight=1)
        destra.columnconfigure(0, weight=1)

        self.box_risultati = tk.Frame(destra, bg="#141428", bd=1, relief="solid")
        self.box_risultati.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(self.box_risultati, text="← Inserisci i dati e clicca ANALIZZA",
                 bg="#141428", fg="#8888aa", font=("Courier", 10)).pack(padx=14, pady=12)

        self.box_grafico = tk.Frame(destra, bg="#141428", bd=1, relief="solid")
        self.box_grafico.grid(row=1, column=0, sticky="nsew")
        self.canvas_grafico = None  # sarà creato quando si clicca ANALIZZA

    def analizza(self):
        """Raccoglie i valori dalla UI, calcola la pick e aggiorna i risultati."""

        # Legge e valida i valori dai campi di input
        statistiche = {}
        for feature, (var, minimo, massimo) in self.input_vars.items():
            try:
                valore = float(var.get())
            except Exception:
                messagebox.showerror("Errore", f"Valore non valido per {feature}")
                return
            if not (minimo <= valore <= massimo):
                messagebox.showerror("Errore", f"{feature}: fuori range [{minimo}, {massimo}]")
                return
            statistiche[feature] = valore

        # Legge gli anni selezionati dalle checkbox
        anni_selezionati = sorted([anno for anno, var in self.anni_vars.items() if var.get()])
        if not anni_selezionati:
            messagebox.showwarning("Attenzione", "Seleziona almeno un anno!")
            return

        nome = self.nome_var.get().strip() or "Giocatore X"

        # Calcolo principale: score → pick
        score     = calcola_score(statistiche, self.df)
        pick_base = score_in_pick(score)

        # Aggiunge un piccolo offset per simulare la variabilità tra anni diversi
        media_globale = self.df["overall_pick"].mean()
        risultati = []
        for anno in anni_selezionati:
            df_anno = self.df[self.df["year"] == anno]
            offset  = (df_anno["overall_pick"].mean() - media_globale) * 0.08 if len(df_anno) > 5 else 0
            pick    = int(np.clip(round(pick_base + offset), 1, 60))
            risultati.append((anno, pick))

        self.mostra_risultati(nome, risultati, score)
        self.mostra_grafico(nome, risultati)

    def mostra_risultati(self, nome, risultati, score):
        """Aggiorna il pannello testuale con i risultati del calcolo."""
        for widget in self.box_risultati.winfo_children():
            widget.destroy()

        picks    = [p for _, p in risultati]
        migliore = min(picks)
        media    = np.mean(picks)

        # Intestazione: nome + score
        intestazione = tk.Frame(self.box_risultati, bg="#141428")
        intestazione.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(intestazione, text=f"🏀 {nome.upper()}", bg="#141428", fg="#f5a623",
                 font=("Georgia", 13, "bold")).pack(side="left")
        tk.Label(intestazione, text=f"  score: {score*100:.0f}° percentile",
                 bg="#141428", fg="#8888aa", font=("Courier", 9)).pack(side="left", pady=3)

        # Pill colorati con pick medio e miglior pick
        contenitore_pill = tk.Frame(self.box_risultati, bg="#141428")
        contenitore_pill.pack(fill="x", padx=14, pady=(0, 8))
        for etichetta, valore, colore in [
            ("PICK MEDIO",   f"#{media:.1f}", "#555577"),
            ("MIGLIOR PICK", f"#{migliore}", colore_pick(migliore)),
        ]:
            pill = tk.Frame(contenitore_pill, bg=colore, padx=10, pady=4)
            pill.pack(side="left", padx=4)
            tk.Label(pill, text=etichetta, bg=colore, fg="white", font=("Courier", 7)).pack()
            tk.Label(pill, text=valore,    bg=colore, fg="white", font=("Courier", 12, "bold")).pack()

        # Lista pick per anno
        lista = tk.Frame(self.box_risultati, bg="#141428")
        lista.pack(fill="x", padx=14, pady=(0, 10))
        for anno, pick in risultati:
            riga = tk.Frame(lista, bg="#1e1e3a", pady=2)
            riga.pack(fill="x", pady=1)
            tk.Label(riga, text=f"  {anno}", bg="#1e1e3a", fg="#8888aa",
                     font=("Courier", 9), width=6).pack(side="left")
            tk.Label(riga, text=f"  #{pick:2d}", bg="#1e1e3a", fg=colore_pick(pick),
                     font=("Courier", 10, "bold"), width=6).pack(side="left")
            zona = next(z for lim, z, _ in ZONE if pick <= lim)
            tk.Label(riga, text=f"  {zona}", bg="#1e1e3a", fg=colore_pick(pick),
                     font=("Courier", 9)).pack(side="left")

    def mostra_grafico(self, nome, risultati):
        """Disegna i due grafici: pick per anno e feature importance."""
        if self.canvas_grafico:
            self.canvas_grafico.get_tk_widget().destroy()

        anni       = [r[0] for r in risultati]
        picks      = [r[1] for r in risultati]
        importanze = self.model.feature_importances_
        etichette  = ["PPG", "REB", "AST", "WS/48", "BPM", "VORP", "FG%", "FT%", "3P%", "Anni"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), facecolor="#141428")
        fig.suptitle(f"{nome} – Mock Draft", color="#f5a623", fontsize=11, fontweight="bold")

        # Grafico 1: pick stimata per anno
        ax1.set_facecolor("#0d0d1a")
        ax1.invert_yaxis()
        ax1.set_ylim(62, 0)
        for limite, zona, colore in ZONE:
            precedente = 0 if limite == 5 else [z[0] for z in ZONE if z[0] < limite][-1]
            ax1.axhspan(precedente, limite, alpha=0.10, color=colore)
        ax1.plot(anni, picks, color="white", lw=1, ls="--", alpha=0.3)
        for anno, pick in zip(anni, picks):
            c = colore_pick(pick)
            ax1.scatter(anno, pick, color=c, s=110, zorder=3, edgecolors="white", lw=0.8)
            ax1.annotate(f"#{pick}", (anno, pick), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, color=c, fontweight="bold")
        ax1.set_xlabel("Anno", color="#8888aa", fontsize=9)
        ax1.set_ylabel("Overall Pick", color="#8888aa", fontsize=9)
        ax1.set_title("Pick stimata per anno", color="white", fontsize=10)
        ax1.tick_params(colors="#8888aa", labelsize=7)
        for lato in ax1.spines.values(): lato.set_color("#2a2a4a")
        ax1.legend(handles=[mpatches.Patch(color=c, label=z) for _, z, c in ZONE],
                   facecolor="#0d0d1a", labelcolor="#8888aa", fontsize=7, loc="lower right")

        # Grafico 2: importanza delle feature (dal Random Forest)
        ax2.set_facecolor("#0d0d1a")
        ordine = np.argsort(importanze)
        colori = ["#f5a623" if importanze[i] > 0.1 else "#3a3a5a" for i in ordine]
        ax2.barh([etichette[i] for i in ordine], [importanze[i] for i in ordine],
                 color=colori, edgecolor="#0d0d1a", height=0.6)
        ax2.set_title("Feature Importance (Random Forest)", color="white", fontsize=10)
        ax2.set_xlabel("Importanza relativa", color="#8888aa", fontsize=9)
        ax2.tick_params(colors="#8888aa", labelsize=8)
        for lato in ax2.spines.values(): lato.set_color("#2a2a4a")

        plt.tight_layout()
        self.canvas_grafico = FigureCanvasTkAgg(fig, master=self.box_grafico)
        self.canvas_grafico.draw()
        self.canvas_grafico.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        plt.close(fig)

if __name__ == "__main__":
    app = App()
    app.mainloop()