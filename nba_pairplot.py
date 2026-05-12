"""
NBA Mock Draft – Random Forest
Inserisci le statistiche di un giocatore ipotetico e scopri
a che pick sarebbe stato draftato nei vari anni scelti da te.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ── CONFIGURAZIONE ────────────────────────────────────────────────────────────
CSV_PATH = "nbaplayersdraft.csv"

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

FEATURE_LABELS = {
    "points_per_game":           "Punti per partita    (es. 18.5)",
    "average_total_rebounds":    "Rimbalzi per partita (es. 7.2)",
    "average_assists":           "Assist per partita   (es. 4.1)",
    "win_shares_per_48_minutes": "Win Shares/48 min    (es. 0.12)",
    "box_plus_minus":            "Box Plus/Minus       (es. 2.5)",
    "value_over_replacement":    "Value Over Replac.   (es. 15.0)",
    "field_goal_percentage":     "FG%                  (es. 0.47)",
    "free_throw_percentage":     "FT%                  (es. 0.78)",
    "3_point_percentage":        "3P%                  (es. 0.36)",
    "years_active":              "Anni in carriera     (es. 8)",
}

# Limiti per validazione input
BOUNDS = {
    "points_per_game":           (0, 40),
    "average_total_rebounds":    (0, 25),
    "average_assists":           (0, 15),
    "win_shares_per_48_minutes": (-0.5, 0.5),
    "box_plus_minus":            (-15, 20),
    "value_over_replacement":    (-30, 100),
    "field_goal_percentage":     (0, 1),
    "free_throw_percentage":     (0, 1),
    "3_point_percentage":        (0, 1),
    "years_active":              (1, 25),
}

# Zone draft
def zona_pick(pick):
    if pick <= 5:   return "🌟 Lottery Top-5"
    if pick <= 14:  return "🔵 Lottery (6-14)"
    if pick <= 30:  return "🟢 Primo giro (15-30)"
    if pick <= 45:  return "🟡 Secondo giro early (31-45)"
    return          "🔴 Secondo giro late (46-60)"

# ── CARICAMENTO E TRAINING ────────────────────────────────────────────────────
print("=" * 60)
print("   NBA MOCK DRAFT – Random Forest Predictor")
print("=" * 60)
print("\n📂 Caricamento dataset...")

df = pd.read_csv(CSV_PATH)
df_clean = df.dropna(subset=FEATURES + ["overall_pick"])

X = df_clean[FEATURES].values
y = df_clean["overall_pick"].values

print(f"   ✅ {len(df_clean)} giocatori caricati ({df_clean['year'].min()}-{df_clean['year'].max()})")
print("\n🌲 Training Random Forest...")
model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X, y)
print("   ✅ Modello pronto!\n")

# ── INPUT GIOCATORE ───────────────────────────────────────────────────────────
print("─" * 60)
print("  INSERISCI LE STATISTICHE DEL GIOCATORE")
print("─" * 60)

player_name = input("\n👤 Nome del giocatore: ").strip() or "Giocatore X"

player_stats = {}
for feat, label in FEATURE_LABELS.items():
    lo, hi = BOUNDS[feat]
    while True:
        try:
            val = float(input(f"   {label}: "))
            if lo <= val <= hi:
                player_stats[feat] = val
                break
            else:
                print(f"   ⚠️  Valore fuori range [{lo}, {hi}], riprova.")
        except ValueError:
            print("   ⚠️  Inserisci un numero valido.")

# ── INPUT ANNI ────────────────────────────────────────────────────────────────
print("\n─" * 60)
print("  INSERISCI GLI ANNI DI DRAFT DA SIMULARE")
print("  (anni disponibili: 1989-2021, separati da virgola)")
print("─" * 60)

anni_validi = set(df["year"].unique())
while True:
    raw = input("\n📅 Anni (es. 1995, 2003, 2010, 2019): ")
    try:
        anni = [int(a.strip()) for a in raw.split(",")]
        anni_non_validi = [a for a in anni if a not in anni_validi]
        if anni_non_validi:
            print(f"   ⚠️  Anni non presenti nel dataset: {anni_non_validi}")
            print(f"   Anni disponibili: {sorted(anni_validi)}")
        else:
            anni = sorted(anni)
            break
    except ValueError:
        print("   ⚠️  Formato non valido. Usa numeri separati da virgola.")

# ── PREDIZIONE ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  RISULTATI PER: {player_name.upper()}")
print("=" * 60)

X_player = np.array([[player_stats[f] for f in FEATURES]])

risultati = []
for anno in anni:
    # Calcoliamo il pick predetto
    pick_raw = model.predict(X_player)[0]

    # Aggiungiamo un effetto "era": pick tende a variare leggermente per anno
    # basandoci sulla media dei pick dello stesso anno nel dataset
    df_anno = df_clean[df_clean["year"] == anno]
    if len(df_anno) > 5:
        media_anno = df_anno["overall_pick"].mean()
        media_globale = df_clean["overall_pick"].mean()
        offset = (media_anno - media_globale) * 0.05
    else:
        offset = 0

    pick_finale = int(np.clip(round(pick_raw + offset), 1, 60))
    zona = zona_pick(pick_finale)
    risultati.append((anno, pick_finale, zona))

    print(f"\n  📅 Anno {anno}")
    print(f"     Pick stimato : #{pick_finale}")
    print(f"     Zona draft   : {zona}")

# ── OUTPUT TESTUALE RIEPILOGO ─────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  RIEPILOGO")
print("─" * 60)
picks = [r[1] for r in risultati]
print(f"  Pick medio  : #{np.mean(picks):.1f}")
print(f"  Pick minore : #{min(picks)} (anno {risultati[picks.index(min(picks))][0]})")
print(f"  Pick maggiore: #{max(picks)} (anno {risultati[picks.index(max(picks))][0]})")

# ── GRAFICO ───────────────────────────────────────────────────────────────────
print("\n📊 Generazione grafico...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="#1a1a2e")
fig.suptitle(f"NBA Mock Draft – {player_name}", fontsize=16,
             fontweight="bold", color="white", y=1.02)

anni_plot = [r[0] for r in risultati]
picks_plot = [r[1] for r in risultati]

# Colori per zona
def colore_pick(pick):
    if pick <= 5:   return "#FFD700"
    if pick <= 14:  return "#4C72B0"
    if pick <= 30:  return "#2ecc71"
    if pick <= 45:  return "#f39c12"
    return "#e74c3c"

colori = [colore_pick(p) for p in picks_plot]

# ── GRAFICO 1: Pick per anno ──────────────────────────────────────────────────
ax1.set_facecolor("#16213e")
ax1.invert_yaxis()
ax1.set_ylim(63, 0)
ax1.set_yticks([1, 5, 14, 30, 45, 60])
ax1.axhspan(0, 5,   alpha=0.08, color="#FFD700")
ax1.axhspan(5, 14,  alpha=0.08, color="#4C72B0")
ax1.axhspan(14, 30, alpha=0.08, color="#2ecc71")
ax1.axhspan(30, 45, alpha=0.08, color="#f39c12")
ax1.axhspan(45, 60, alpha=0.08, color="#e74c3c")

ax1.plot(anni_plot, picks_plot, color="white", linewidth=1.5,
         linestyle="--", alpha=0.4, zorder=1)
for anno, pick, colore in zip(anni_plot, picks_plot, colori):
    ax1.scatter(anno, pick, color=colore, s=180, zorder=3,
                edgecolors="white", linewidths=1.5)
    ax1.annotate(f"#{pick}", (anno, pick),
                 textcoords="offset points", xytext=(0, 10),
                 ha="center", fontsize=10, color=colore, fontweight="bold")

ax1.set_xlabel("Anno di Draft", color="white", fontsize=12)
ax1.set_ylabel("Overall Pick", color="white", fontsize=12)
ax1.set_title("Pick stimato per anno", color="white", fontsize=13, pad=10)
ax1.tick_params(colors="white")
for spine in ax1.spines.values():
    spine.set_color("#444")

# Legenda zone
legend_items = [
    mpatches.Patch(color="#FFD700", label="Top-5"),
    mpatches.Patch(color="#4C72B0", label="Lottery 6-14"),
    mpatches.Patch(color="#2ecc71", label="1° giro 15-30"),
    mpatches.Patch(color="#f39c12", label="2° giro 31-45"),
    mpatches.Patch(color="#e74c3c", label="2° giro 46-60"),
]
ax1.legend(handles=legend_items, loc="lower right",
           facecolor="#1a1a2e", labelcolor="white", fontsize=8)

# ── GRAFICO 2: Feature importance + profilo giocatore ────────────────────────
ax2.set_facecolor("#16213e")
importances = model.feature_importances_
feat_labels_short = ["PPG", "REB", "AST", "WS/48", "BPM", "VORP", "FG%", "FT%", "3P%", "Anni"]
sorted_idx = np.argsort(importances)

bars = ax2.barh([feat_labels_short[i] for i in sorted_idx],
                [importances[i] for i in sorted_idx],
                color="#4C72B0", edgecolor="#1a1a2e", height=0.6)

# Evidenzia le feature più importanti
for i, bar in enumerate(bars):
    if importances[sorted_idx[i]] > 0.1:
        bar.set_color("#FFD700")

ax2.set_title("Importanza delle feature (Random Forest)",
              color="white", fontsize=13, pad=10)
ax2.set_xlabel("Importanza relativa", color="white", fontsize=11)
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_color("#444")

# Valori del giocatore inseriti
y_pos = 0.98
ax2.text(0.62, y_pos, f"Profilo: {player_name}", transform=ax2.transAxes,
         color="white", fontsize=9, fontweight="bold", va="top")
stats_testo = [
    f"PPG: {player_stats['points_per_game']:.1f}",
    f"REB: {player_stats['average_total_rebounds']:.1f}",
    f"AST: {player_stats['average_assists']:.1f}",
    f"BPM: {player_stats['box_plus_minus']:.1f}",
    f"WS/48: {player_stats['win_shares_per_48_minutes']:.3f}",
]
for i, testo in enumerate(stats_testo):
    ax2.text(0.62, y_pos - 0.07*(i+1), testo, transform=ax2.transAxes,
             color="#aaaaaa", fontsize=8, va="top")

plt.tight_layout()
output_img = "nba_mock_draft_result.png"
plt.savefig(output_img, dpi=150, bbox_inches="tight",
            facecolor="#1a1a2e")
print(f"   ✅ Grafico salvato: {output_img}")
plt.show()

print("\n✅ Completato!\n")