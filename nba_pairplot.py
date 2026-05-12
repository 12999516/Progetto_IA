"""
NBA Mock Draft – GUI (fixed scrollable panel)
"""

import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor

# ── COLORI ────────────────────────────────────────────────────────────────────
BG_DARK  = "#0d0d1a"
BG_CARD  = "#141428"
BG_INPUT = "#1e1e3a"
ACCENT   = "#f5a623"
ACCENT2  = "#e74c3c"
TEXT_W   = "#f0f0f0"
TEXT_G   = "#8888aa"
TEXT_GOLD= "#FFD700"
BORDER   = "#2a2a4a"
GREEN    = "#2ecc71"
BLUE     = "#4C72B0"

FEATURES = [
    "points_per_game","average_total_rebounds","average_assists",
    "win_shares_per_48_minutes","box_plus_minus","value_over_replacement",
    "field_goal_percentage","free_throw_percentage","3_point_percentage","years_active",
]

FIELD_INFO = [
    ("points_per_game",           "PPG",   "Punti per partita",         0,   40,  18.5),
    ("average_total_rebounds",    "REB",   "Rimbalzi per partita",       0,   25,   7.2),
    ("average_assists",           "AST",   "Assist per partita",         0,   15,   4.1),
    ("win_shares_per_48_minutes", "WS/48", "Win Shares per 48 min",    -0.5, 0.5,  0.12),
    ("box_plus_minus",            "BPM",   "Box Plus/Minus",           -15,  20,   2.5),
    ("value_over_replacement",    "VORP",  "Value Over Replacement",   -30, 100,  15.0),
    ("field_goal_percentage",     "FG%",   "Field Goal % (es. 0.47)",    0,   1,   0.47),
    ("free_throw_percentage",     "FT%",   "Free Throw % (es. 0.78)",    0,   1,   0.78),
    ("3_point_percentage",        "3P%",   "3 Point % (es. 0.36)",       0,   1,   0.36),
    ("years_active",              "ANNI",  "Anni in carriera NBA",        1,  25,   8),
]

ANNI_DISPONIBILI = list(range(1989, 2022))

def zona_pick(pick):
    if pick <= 5:  return ("🌟 Top-5",         TEXT_GOLD)
    if pick <= 14: return ("🔵 Lottery 6-14",  BLUE)
    if pick <= 30: return ("🟢 1° Giro 15-30", GREEN)
    if pick <= 45: return ("🟡 2° Giro 31-45", "#f39c12")
    return               ("🔴 2° Giro 46-60",  "#e74c3c")

def colore_pick(pick):
    if pick <= 5:  return TEXT_GOLD
    if pick <= 14: return BLUE
    if pick <= 30: return GREEN
    if pick <= 45: return "#f39c12"
    return "#e74c3c"

def train_model():
    df = pd.read_csv("nbaplayersdraft.csv")
    df_clean = df.dropna(subset=FEATURES + ["overall_pick"])
    X = df_clean[FEATURES].values
    y = df_clean["overall_pick"].values
    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, df_clean


class ScrollableFrame(tk.Frame):
    def __init__(self, parent, bg, **kwargs):
        super().__init__(parent, bg=bg, **kwargs)
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.inner = tk.Frame(self._canvas, bg=bg)
        self.inner.bind("<Configure>", lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self._canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        self._canvas.bind("<MouseWheel>", self._on_scroll)
        self.inner.bind("<MouseWheel>", self._on_scroll)

    def _on_scroll(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def bind_scroll_recursive(self, widget):
        widget.bind("<MouseWheel>", self._on_scroll)
        for child in widget.winfo_children():
            self.bind_scroll_recursive(child)


class NBADraftApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NBA Mock Draft – Random Forest")
        self.configure(bg=BG_DARK)
        self.geometry("1150x820")
        self.minsize(950, 700)

        lbl = tk.Label(self, text="⏳ Caricamento modello Random Forest...",
                       bg=BG_DARK, fg=ACCENT, font=("Courier", 13))
        lbl.pack(pady=60)
        self.update()
        self.model, self.df = train_model()
        lbl.destroy()

        self._build_ui()

    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill="x", padx=20, pady=(12, 4))
        tk.Label(hdr, text="🏀 NBA MOCK DRAFT", bg=BG_DARK, fg=ACCENT,
                 font=("Georgia", 20, "bold")).pack(side="left")
        tk.Label(hdr, text="  powered by Random Forest", bg=BG_DARK, fg=TEXT_G,
                 font=("Courier", 9)).pack(side="left", pady=5)

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 12))
        body.columnconfigure(0, weight=0, minsize=370)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_right(body)

    def _build_left(self, parent):
        outer = tk.Frame(parent, bg=BG_CARD,
                         highlightthickness=1, highlightbackground=BORDER)
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        self.sf = ScrollableFrame(outer, bg=BG_CARD)
        self.sf.grid(row=0, column=0, sticky="nsew")
        inner = self.sf.inner

        tk.Label(inner, text="PROFILO GIOCATORE", bg=BG_CARD, fg=ACCENT,
                 font=("Courier", 11, "bold")).pack(pady=(14, 6))

        # Nome
        nf = tk.Frame(inner, bg=BG_CARD)
        nf.pack(fill="x", padx=14, pady=(0, 8))
        tk.Label(nf, text="👤 Nome giocatore", bg=BG_CARD, fg=TEXT_G,
                 font=("Courier", 8)).pack(anchor="w")
        self.name_var = tk.StringVar(value="Giocatore X")
        tk.Entry(nf, textvariable=self.name_var, bg=BG_INPUT, fg=TEXT_W,
                 insertbackground=ACCENT, relief="flat", font=("Courier", 11),
                 bd=0, highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT).pack(fill="x", ipady=6)

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=14, pady=6)
        tk.Label(inner, text="📊 STATISTICHE", bg=BG_CARD, fg=TEXT_G,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=14)

        self.stat_vars = {}
        for feat, abbr, label, lo, hi, default in FIELD_INFO:
            row = tk.Frame(inner, bg=BG_CARD)
            row.pack(fill="x", padx=14, pady=3)
            tk.Label(row, text=abbr, width=6, bg=ACCENT2, fg="white",
                     font=("Courier", 8, "bold"), relief="flat").pack(
                         side="left", ipady=4, padx=(0, 8))
            col = tk.Frame(row, bg=BG_CARD)
            col.pack(side="left", fill="x", expand=True)
            tk.Label(col, text=f"{label}  [{lo}÷{hi}]", bg=BG_CARD, fg=TEXT_G,
                     font=("Courier", 7)).pack(anchor="w")
            var = tk.DoubleVar(value=default)
            self.stat_vars[feat] = (var, lo, hi)
            tk.Entry(col, textvariable=var, bg=BG_INPUT, fg=TEXT_W,
                     insertbackground=ACCENT, relief="flat", font=("Courier", 11),
                     bd=0, highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=ACCENT, width=14).pack(anchor="w", ipady=4)

        # Anni
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=14, pady=8)
        tk.Label(inner, text="📅 ANNI DA SIMULARE", bg=BG_CARD, fg=TEXT_G,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=14)

        br = tk.Frame(inner, bg=BG_CARD)
        br.pack(fill="x", padx=14, pady=(4, 4))
        tk.Button(br, text="Seleziona tutti", command=self._sel_all,
                  bg=BG_INPUT, fg=TEXT_G, relief="flat",
                  font=("Courier", 8), cursor="hand2").pack(side="left", padx=(0, 6))
        tk.Button(br, text="Deseleziona tutti", command=self._sel_none,
                  bg=BG_INPUT, fg=TEXT_G, relief="flat",
                  font=("Courier", 8), cursor="hand2").pack(side="left")

        gf = tk.Frame(inner, bg=BG_CARD)
        gf.pack(fill="x", padx=14, pady=(2, 6))
        self.year_vars = {}
        for i, anno in enumerate(ANNI_DISPONIBILI):
            var = tk.BooleanVar(value=False)
            self.year_vars[anno] = var
            cb = tk.Checkbutton(gf, text=str(anno), variable=var,
                           bg=BG_CARD, fg=TEXT_W, selectcolor=BG_INPUT,
                           activebackground=BG_CARD, activeforeground=ACCENT,
                           font=("Courier", 8), relief="flat", bd=0)
            cb.grid(row=i // 6, column=i % 6, sticky="w", padx=2, pady=2)

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=14, pady=8)

        tk.Button(inner, text="▶  ANALIZZA DRAFT",
                  command=self._run,
                  bg=ACCENT, fg=BG_DARK,
                  font=("Courier", 13, "bold"),
                  relief="flat", cursor="hand2",
                  activebackground="#e6951a",
                  activeforeground=BG_DARK).pack(
                      fill="x", padx=14, pady=(0, 18), ipady=12)

        # Bind scroll su tutti i widget interni
        self.after(200, lambda: self.sf.bind_scroll_recursive(inner))

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self.summary_frame = tk.Frame(right, bg=BG_CARD,
                                      highlightthickness=1, highlightbackground=BORDER)
        self.summary_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tk.Label(self.summary_frame,
                 text="← Inserisci i dati e clicca ANALIZZA DRAFT",
                 bg=BG_CARD, fg=TEXT_G, font=("Courier", 10)).pack(
                     padx=15, pady=14, anchor="w")

        self.chart_frame = tk.Frame(right, bg=BG_CARD,
                                    highlightthickness=1, highlightbackground=BORDER)
        self.chart_frame.grid(row=1, column=0, sticky="nsew")
        self.canvas_widget = None

    def _sel_all(self):
        for v in self.year_vars.values(): v.set(True)

    def _sel_none(self):
        for v in self.year_vars.values(): v.set(False)

    def _run(self):
        player_stats = {}
        for feat, (var, lo, hi) in self.stat_vars.items():
            try:
                val = var.get()
            except Exception:
                messagebox.showerror("Errore", f"Valore non valido per {feat}")
                return
            if not (lo <= val <= hi):
                messagebox.showerror("Errore",
                    f"Valore fuori range per {feat}\nDeve essere tra {lo} e {hi}")
                return
            player_stats[feat] = val

        anni = sorted([a for a, v in self.year_vars.items() if v.get()])
        if not anni:
            messagebox.showwarning("Attenzione", "Seleziona almeno un anno!")
            return

        name = self.name_var.get().strip() or "Giocatore X"
        X_p = np.array([[player_stats[f] for f in FEATURES]])
        pick_base = self.model.predict(X_p)[0]
        media_glob = self.df["overall_pick"].mean()

        risultati = []
        for anno in anni:
            df_a = self.df[self.df["year"] == anno]
            offset = (df_a["overall_pick"].mean() - media_glob) * 0.05 if len(df_a) > 5 else 0
            pick = int(np.clip(round(pick_base + offset), 1, 60))
            risultati.append((anno, pick))

        self._show_summary(name, risultati)
        self._show_chart(name, risultati)

    def _show_summary(self, name, risultati):
        for w in self.summary_frame.winfo_children():
            w.destroy()

        picks = [r[1] for r in risultati]
        best = min(picks)
        worst = max(picks)
        best_anno = risultati[picks.index(best)][0]
        worst_anno = risultati[picks.index(worst)][0]
        zona_label, zona_color = zona_pick(best)

        tk.Label(self.summary_frame, text=f"🏀  {name.upper()}",
                 bg=BG_CARD, fg=ACCENT,
                 font=("Georgia", 13, "bold")).pack(anchor="w", padx=14, pady=(10, 4))

        pills = tk.Frame(self.summary_frame, bg=BG_CARD)
        pills.pack(fill="x", padx=14, pady=(0, 8))

        def pill(parent, lbl, val, color):
            f = tk.Frame(parent, bg=color, padx=10, pady=5)
            f.pack(side="left", padx=4)
            tk.Label(f, text=lbl, bg=color, fg="white", font=("Courier", 7)).pack()
            tk.Label(f, text=val, bg=color, fg="white", font=("Courier", 11, "bold")).pack()

        pill(pills, "PICK MEDIO",            f"#{np.mean(picks):.1f}",   "#555577")
        pill(pills, f"BEST ({best_anno})",   f"#{best}",                 colore_pick(best))
        pill(pills, f"WORST ({worst_anno})", f"#{worst}",                colore_pick(worst))
        pill(pills, "ZONA BEST",             zona_label.split(" ", 1)[1], zona_color)

        lf = tk.Frame(self.summary_frame, bg=BG_CARD)
        lf.pack(fill="x", padx=14, pady=(0, 10))
        for anno, pick in risultati:
            zl, zc = zona_pick(pick)
            r = tk.Frame(lf, bg=BG_INPUT, pady=2)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=f"  {anno}", bg=BG_INPUT, fg=TEXT_G,
                     font=("Courier", 9), width=6).pack(side="left")
            tk.Label(r, text=f"  #{pick:2d}", bg=BG_INPUT, fg=colore_pick(pick),
                     font=("Courier", 10, "bold"), width=6).pack(side="left")
            tk.Label(r, text=f"  {zl}", bg=BG_INPUT, fg=zc,
                     font=("Courier", 9)).pack(side="left")

    def _show_chart(self, name, risultati):
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()

        anni_p = [r[0] for r in risultati]
        picks_p = [r[1] for r in risultati]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), facecolor=BG_CARD)
        fig.suptitle(f"{name} – Simulazione Draft",
                     color=ACCENT, fontsize=12, fontweight="bold")

        ax1.set_facecolor(BG_DARK)
        ax1.invert_yaxis()
        ax1.set_ylim(63, 0)
        ax1.set_yticks([1, 5, 14, 30, 45, 60])
        ax1.axhspan(0,  5,  alpha=0.12, color=TEXT_GOLD)
        ax1.axhspan(5,  14, alpha=0.12, color=BLUE)
        ax1.axhspan(14, 30, alpha=0.12, color=GREEN)
        ax1.axhspan(30, 45, alpha=0.12, color="#f39c12")
        ax1.axhspan(45, 60, alpha=0.12, color="#e74c3c")
        ax1.plot(anni_p, picks_p, color="white", lw=1, ls="--", alpha=0.3)
        for a, p in zip(anni_p, picks_p):
            c = colore_pick(p)
            ax1.scatter(a, p, color=c, s=120, zorder=3,
                        edgecolors="white", linewidths=1)
            ax1.annotate(f"#{p}", (a, p), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8,
                         color=c, fontweight="bold")
        ax1.set_xlabel("Anno", color=TEXT_G, fontsize=9)
        ax1.set_ylabel("Overall Pick", color=TEXT_G, fontsize=9)
        ax1.set_title("Pick stimato per anno", color=TEXT_W, fontsize=10)
        ax1.tick_params(colors=TEXT_G, labelsize=7)
        for s in ax1.spines.values(): s.set_color(BORDER)
        ax1.legend(handles=[
            mpatches.Patch(color=TEXT_GOLD, label="Top-5"),
            mpatches.Patch(color=BLUE,      label="Lottery 6-14"),
            mpatches.Patch(color=GREEN,     label="1° giro 15-30"),
            mpatches.Patch(color="#f39c12", label="2° giro 31-45"),
            mpatches.Patch(color="#e74c3c", label="2° giro 46-60"),
        ], loc="lower right", facecolor=BG_DARK,
           labelcolor=TEXT_G, fontsize=7, framealpha=0.8)

        ax2.set_facecolor(BG_DARK)
        imp = self.model.feature_importances_
        lbs = ["PPG","REB","AST","WS/48","BPM","VORP","FG%","FT%","3P%","Anni"]
        idx = np.argsort(imp)
        ax2.barh([lbs[i] for i in idx], [imp[i] for i in idx],
                 color=[ACCENT if imp[i] > 0.1 else "#3a3a5a" for i in idx],
                 edgecolor=BG_DARK, height=0.6)
        ax2.set_title("Feature Importance", color=TEXT_W, fontsize=10)
        ax2.set_xlabel("Importanza", color=TEXT_G, fontsize=9)
        ax2.tick_params(colors=TEXT_G, labelsize=8)
        for s in ax2.spines.values(): s.set_color(BORDER)

        plt.tight_layout()
        self.canvas_widget = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        plt.close(fig)


if __name__ == "__main__":
    app = NBADraftApp()
    app.mainloop()