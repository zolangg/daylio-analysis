import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess


# --- SCHWELLENWERT-FUNKTIONEN ---
def varianz_warnschwelle(win): return max(0.2, 0.4 - 0.0015 * win)


def varianz_kritschwelle(win): return max(0.3, 0.5 - 0.001 * win)


def autokorr_warnschwelle(win): return 0.45 - 0.0005 * win


def autokorr_kritschwelle(win): return 0.65 - 0.0003 * win


def shannon_warnschwelle(win): return 1.3 + 0.001 * win


def shannon_kritschwelle(win): return 1.6 + 0.001 * win


def apen_warnschwelle(win): return 0.5 + 0.001 * win


def apen_kritschwelle(win): return 0.7 + 0.0015 * win


def shannon_entropy(sequence):
    values, counts = np.unique(sequence, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return entropy


def approximate_entropy(U, m=2, r=0.2):
    U = np.array(U)
    N = len(U)
    if N <= m + 1:
        return np.nan

    def _phi(m):
        X = np.array([U[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r, axis=0) / (N - m + 1.0)
        return np.sum(np.log(C + 1e-12)) / (N - m + 1.0)

    return abs(_phi(m) - _phi(m + 1))


def fig1_to_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf


st.set_page_config(layout="wide", page_title="Daylio Stimmungsanalyse")
st.title("Daylio Stimmungsanalyse")

st.write("""
Lade deinen Daylio-Export (CSV) hoch und erhalte die wichtigsten Visualisierungen:
- Häufigkeitsverteilung und Mood Zeitverlauf
- Stimmungsglättung (Savitzky-Golay, LOESS)
- Rolling Varianz & Autokorrelation (Frühwarnsignale)
- Entropie-Maße als Stabilitätsindikator
- Identifikation von Tagen mit intra-täglichen Mixed States (keine Inter-Mixed, kein Markov)
""")

uploaded_file = st.file_uploader("Daylio CSV-Datei hochladen", type=["csv"])

if uploaded_file:
    # --- DATEN EINLESEN UND MAPPEN ---
    df = pd.read_csv(uploaded_file)
    mood_map = {
        'Super Low': 1,
        'Low': 2,
        'Euthym': 3,
        'High': 4,
        'Super High': 5,
    }
    df['full_date'] = pd.to_datetime(df['full_date'])
    df = df.sort_values('full_date')
    df['Stimmungswert'] = df['mood'].map(mood_map)

    # --- Mixed-State Grundlage: Tagesweise alle Mood-Listen sammeln ---
    daily_moods = df.groupby('full_date')['Stimmungswert'].agg(list)
    intra_var = daily_moods.apply(np.std)
    intra_range = daily_moods.apply(lambda x: max(x) - min(x) if len(x) > 1 else 0)
    mixed_intra = (intra_range >= 2) | (intra_var >= 1)

    df_intraday = pd.DataFrame({
        'Datum': daily_moods.index,
        'Mood_List': daily_moods.values,
        'Intra_Range': intra_range.values,
        'Intra_Std': intra_var.values,
        'Mixed_IntraDay': mixed_intra.values,
    })

    # --- Tagesmittel für klassische Plots ---
    df_tagesmittel = df.groupby('full_date')['Stimmungswert'].mean().reset_index()
    df_tagesmittel = df_tagesmittel.sort_values('full_date')
    df_tagesmittel['Datum'] = df_tagesmittel['full_date']

    # --- Sidebar für Frühwarnsignale/Entropie ---
    st.sidebar.header("Frühwarnsignal-Einstellungen")
    loess_frac = st.sidebar.slider("LOESS Glättung (Fraktion)", 0.05, 0.20, 0.08)
    sg_win = st.sidebar.slider("Savitzky-Golay Fenster (Tage)", min_value=7, max_value=61, value=31, step=2)
    win = st.sidebar.slider("Rolling-Fenster (Varianz/Autokorr)", min_value=7, max_value=180, value=90, step=1)
    entropy_win = st.sidebar.slider("Entropie-Fenster (Tage)", min_value=7, max_value=180, value=60, step=1)

    # --- Frühwarnsignale ---
    df_tagesmittel['Varianz'] = df_tagesmittel['Stimmungswert'].rolling(window=win).var()
    df_tagesmittel['Autokorr'] = df_tagesmittel['Stimmungswert'].rolling(window=win).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False)

    # --- Entropie-Berechnung ---
    shannon_entropies, apen_entropies = [], []
    data = df_tagesmittel['Stimmungswert'].values
    for i in range(len(data)):
        if i < entropy_win - 1:
            shannon_entropies.append(np.nan)
            apen_entropies.append(np.nan)
        else:
            window = data[i - entropy_win + 1: i + 1]
            shannon_entropies.append(shannon_entropy(window))
            std_win = np.std(window)
            apen_entropies.append(approximate_entropy(window, m=2, r=0.2 * std_win if std_win > 0 else 0.2))
    df_tagesmittel['Shannon Entropy'] = shannon_entropies
    df_tagesmittel['Approximate Entropy'] = apen_entropies

    # --- WARN-POPUP (Kritische Schwellen überschritten?) ---
    warnungen = []
    akt_var = df_tagesmittel['Varianz'].iloc[-1]
    akt_aut = df_tagesmittel['Autokorr'].iloc[-1]
    akt_shannon = df_tagesmittel['Shannon Entropy'].iloc[-1]
    akt_apen = df_tagesmittel['Approximate Entropy'].iloc[-1]
    if akt_var > varianz_kritschwelle(win):
        warnungen.append(f"**Varianz kritisch:** {akt_var:.2f} > {varianz_kritschwelle(win):.2f} (Instabilität!)")
    if akt_aut > autokorr_kritschwelle(win):
        warnungen.append(
            f"**Autokorrelation kritisch:** {akt_aut:.2f} > {autokorr_kritschwelle(win):.2f} (Persistente Stimmungslage/Frühwarnsignal)")
    if akt_shannon > shannon_kritschwelle(entropy_win):
        warnungen.append(
            f"**Shannon Entropie kritisch:** {akt_shannon:.2f} > {shannon_kritschwelle(entropy_win):.2f} (Stimmung sehr chaotisch)")
    if akt_apen > apen_kritschwelle(entropy_win):
        warnungen.append(
            f"**Approximate Entropy kritisch:** {akt_apen:.2f} > {apen_kritschwelle(entropy_win):.2f} (Unregelmäßiger Verlauf)")
    if warnungen:
        st.error("🚨 **KRITISCHE WARNUNG:**\n\n" + "\n\n".join(warnungen))

    # --- Häufigkeitsverteilung (0.5 Schritte) ---
    st.subheader("Häufigkeitsverteilung der Mood-Tageswerte")
    bins = np.arange(1, 5.6, 0.5)
    hist, bin_edges = np.histogram(df_tagesmittel['Stimmungswert'], bins=bins)
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    ax_hist.bar(bin_edges[:-1] + 0.25, hist, width=0.5, color='skyblue', edgecolor='black')
    ax_hist.set_xticks(bin_edges[:-1] + 0.25)
    ax_hist.set_xticklabels([f"{b:.1f}" for b in bin_edges[:-1]])
    ax_hist.set_xlabel("Mood-Wert (0.5 Stufen)")
    ax_hist.set_ylabel("Tage")
    ax_hist.set_title("Häufigkeitsverteilung der Mood-Tageswerte")
    st.pyplot(fig_hist)
    st.download_button(
        "Download Häufigkeitsverteilung als PNG",
        data=fig1_to_bytes(fig_hist),
        file_name="haeufigkeitsverteilung.png"
    )
    st.caption("""
        Interpretation: Das Histogramm zeigt die Verteilung der täglichen Stimmungsmittelwerte in 0.5er-Schritten.
        So erkennst du, welche Stimmungskategorien besonders häufig vertreten sind und ob es Tendenzen zu einem bestimmten Pol (depressiv, euthym, hypomanisch/manisch) gibt.
        Eine Häufung in den extremen Bereichen kann für eine erhöhte Instabilität oder Polymodalität sprechen.
    """)

    # --- Mood Zeitverlauf (farbcodiert nach 0.5 Stufen) ---
    st.subheader("Mood Zeitverlauf")
    mood_vals = df_tagesmittel['Stimmungswert'].values
    dates = df_tagesmittel['Datum'].values


    def color_for_mood(val):
        if val <= 1.5:
            return '#8e44ad'
        elif val <= 2.0:
            return '#5e72b0'
        elif val <= 2.5:
            return '#3498db'
        elif val <= 3.0:
            return '#f1c40f'
        elif val <= 3.5:
            return '#e67e22'
        elif val <= 4.0:
            return '#e74c3c'
        elif val <= 4.5:
            return '#c0392b'
        else:
            return '#900c3f'


    colors = [color_for_mood(v) for v in mood_vals]
    fig_mood, ax_mood = plt.subplots(figsize=(14, 5))
    ax_mood.scatter(dates, mood_vals, c=colors, s=30, label="Mood-Wert")
    ax_mood.plot(dates, mood_vals, color='gray', alpha=0.4, linewidth=1)
    for y in np.arange(1.5, 5.1, 0.5):
        ax_mood.axhline(y, color='lightgray', linestyle='--', linewidth=0.5)
    ax_mood.set_ylabel("Mood (1=Super Low ... 5=Super High)")
    ax_mood.set_xlabel("Datum")
    ax_mood.set_title("Mood Zeitverlauf")
    st.pyplot(fig_mood)
    st.download_button(
        "Download Mood Zeitverlauf als PNG",
        data=fig1_to_bytes(fig_mood),
        file_name="mood_zeitverlauf.png"
    )
    st.caption("""
        Interpretation: Der Verlauf zeigt die Entwicklung der Stimmungsmittelwerte im Zeitverlauf. 
        Farbige Punkte markieren die Stimmungskategorie je Tag. Stabile Phasen, plötzliche Sprünge oder längere Extrembereiche werden sofort sichtbar.
        Perioden mit schnellen Wechseln, Clustern oder Plateaus können auf gemischte oder instabile Verläufe hindeuten.
    """)

    # --- Stimmungsglättung ---
    st.subheader("Stimmungsglättung")
    raw = df_tagesmittel['Stimmungswert'].values
    x = np.arange(len(raw))
    try:
        if len(raw) >= sg_win:
            sg = savgol_filter(raw, window_length=sg_win, polyorder=3)
        else:
            sg = np.full_like(raw, np.nan)
    except Exception:
        sg = np.full_like(raw, np.nan)
    loess_curve = lowess(raw, x, frac=0.08, return_sorted=False)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df_tagesmittel['Datum'], raw, color='gold', alpha=0.2, label="Tagesmittel (roh)")
    ax2.plot(df_tagesmittel['Datum'], sg, color='orange', linewidth=2, label="Savitzky-Golay")
    ax2.plot(df_tagesmittel['Datum'], loess_curve, color='crimson', linewidth=2, label="LOESS")
    for y in np.arange(1.5, 5.1, 0.5):
        ax2.axhline(y, color='lightgray', linestyle='--', linewidth=0.7)
    ax2.axhline(2.5, color='grey', linestyle='--', linewidth=1, alpha=0.6)
    ax2.text(df_tagesmittel['Datum'].iloc[5], 2.5 + 0.05, "Schwelle Depression (2.5)", color='grey', fontsize=9,
             va='bottom')
    ax2.axhline(3.5, color='grey', linestyle='--', linewidth=1, alpha=0.6)
    ax2.text(df_tagesmittel['Datum'].iloc[5], 3.5 + 0.05, "Schwelle Hypomanie (3.5)", color='grey', fontsize=9,
             va='bottom')
    ax2.set_title("Stimmungsglättung")
    ax2.set_xlabel("Datum")
    ax2.set_ylabel("Stimmungswert")
    ax2.legend(loc='upper left')
    st.pyplot(fig2)
    st.download_button(
        "Download Stimmungsglättung als PNG",
        data=fig1_to_bytes(fig2),
        file_name="stimmungsglaettung.png"
    )
    st.caption("""
        Interpretation: Die geglätteten Kurven (Savitzky-Golay, LOESS) machen langfristige Trends und langsame Veränderungen sichtbar. 
        Saisonalitäten, Phasenübergänge oder längerfristige Instabilität werden so erkennbar, auch wenn Tageswerte stark schwanken.
        Eine anhaltend hohe oder niedrige Linie signalisiert längere Stimmungsverschiebungen.
    """)

    # --- Frühwarnsignale: Varianz und Autokorrelation ---
    st.subheader("Überlagerte Frühwarnsignale: Varianz vs. Autokorrelation")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df_tagesmittel['Datum'], df_tagesmittel['Varianz'], color='gold', label=f'Rolling Varianz ({win} Tage)')
    ax1.plot(df_tagesmittel['Datum'], df_tagesmittel['Autokorr'], color='orange',
             label=f'Rolling Autokorrelation (Lag 1, {win} Tage)')
    ax1.axhline(varianz_warnschwelle(win), color='grey', linestyle='--', alpha=0.4, linewidth=1)
    ax1.axhline(varianz_kritschwelle(win), color='grey', linestyle=':', alpha=0.4, linewidth=1)
    ax1.axhline(autokorr_warnschwelle(win), color='grey', linestyle='--', alpha=0.4, linewidth=1)
    ax1.axhline(autokorr_kritschwelle(win), color='grey', linestyle=':', alpha=0.4, linewidth=1)
    ax1.set_title("Überlagerte Frühwarnsignale: Varianz vs. Autokorrelation")
    ax1.set_xlabel("Datum")
    ax1.set_ylabel("Wert")
    ax1.legend(loc='upper left')
    st.pyplot(fig1)
    st.download_button("Download Frühwarnsignale als PNG", data=fig1_to_bytes(fig1), file_name="fruehwarnsignale.png")
    st.caption("""
        Interpretation: Varianz zeigt, wie stark die Stimmung schwankt – hohe Werte deuten auf größere Labilität oder viele Extreme hin.
        Autokorrelation misst die Ähnlichkeit aufeinanderfolgender Tage – ein plötzlicher Anstieg kann auf den Beginn einer Episode hindeuten.
        Warn- und Kritisch-Linien basieren auf typischen Schwellenwerten aus Studien zur Früherkennung von Episoden oder Instabilität.
    """)

    # --- Entropie ---
    st.subheader("Shannon Entropie & Approximate Entropie (Stabilität der Stimmung)")
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(df_tagesmittel['Datum'], df_tagesmittel['Shannon Entropy'], label='Shannon Entropie', color='blue')
    ax4.plot(df_tagesmittel['Datum'], df_tagesmittel['Approximate Entropy'], label='Approximate Entropy', color='red')
    ax4.axhline(shannon_warnschwelle(entropy_win), color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax4.axhline(shannon_kritschwelle(entropy_win), color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax4.axhline(apen_warnschwelle(entropy_win), color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax4.axhline(apen_kritschwelle(entropy_win), color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax4.set_title(f"Stabilität der Stimmung: Shannon & Approximate Entropy ({entropy_win}-Tage-Fenster)")
    ax4.set_xlabel("Datum")
    ax4.set_ylabel("Entropie")
    ax4.legend(loc='upper left')
    st.pyplot(fig4)
    st.download_button("Download Entropie-Plot als PNG", data=fig1_to_bytes(fig4), file_name="entropie.png")
    st.caption("""
        Interpretation: Die Entropiewerte messen die Unvorhersehbarkeit und Komplexität deiner Stimmung im Zeitfenster.
        Hohe Werte bedeuten chaotische, schwer vorhersagbare Phasen – niedrige Werte stehen für Gleichförmigkeit und Stabilität.
        Kritische Schwellen sind an Early-Warning-Signal-Studien und EMA-Forschung angelehnt.
    """)

    # --- Intra-tägliche Mixed-State-Phasen ---
    st.subheader("Intra-tägliche Mixed-State-Phasen")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df_intraday['Datum'], [np.mean(m) for m in df_intraday['Mood_List']], color='lightgray', label='Tagesmittel')
    ax.scatter(
        df_intraday['Datum'][df_intraday['Mixed_IntraDay']],
        [np.mean(m) for i, m in enumerate(df_intraday['Mood_List']) if df_intraday['Mixed_IntraDay'].iloc[i]],
        color='crimson', label='Intra-täglicher Mixed-State', s=40
    )
    ax.set_ylabel('Mood')
    ax.set_title('Tage mit intra-täglichen Mixed States')
    ax.legend(loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    st.download_button(
        "Download Intra-tägliche Mixed-State Zeitreihe",
        data=fig1_to_bytes(fig),
        file_name="intra_mixed_state_zeitreihe.png"
    )
    st.caption("""
        Interpretation: Dieser Plot zeigt alle Tage, an denen mindestens zwei sehr unterschiedliche Stimmungseinträge (Range ≥2 Mood-Punkte ODER Standardabweichung ≥1) innerhalb eines Tages gemessen wurden. 
        Diese Tage gelten als besonders instabil oder gemischt – sie können subklinische oder klinische Mischzustände markieren, wie sie auch in der EMA- und Verlaufsforschung zur Bipolarität beschrieben werden.
    """)

    # --- Statistik und Tabelle ---
    n_intra = int(df_intraday['Mixed_IntraDay'].sum())
    st.markdown(f"""
**Statistik Intra-tägliche Mixed-State-Tage:**  
- Intra-tägliche Mixed-States: **{n_intra}** von insgesamt **{len(df_intraday)}** Tagen
""")

    # --- Label-Analyse-Block (nur Heatmap, separater Abschnitt) ---
        # --- Label-Analyse-Block (nur Heatmap, kein Filter) ---
    st.subheader("Label-Analyse (Heatmap)")

    if 'activities' in df.columns:
        df['Label_List'] = df['activities'].fillna('').apply(lambda x: [a.strip() for a in x.split('|')] if x else [])
        # Heatmap Labels vs. Mood
        mood_bins = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        df_expl = df.explode('Label_List')
        df_expl = df_expl[df_expl['Label_List'] != '']
        df_expl['Mood_Bin'] = pd.cut(df_expl['Stimmungswert'], bins=mood_bins, include_lowest=True, right=False)
        heatmap_data = pd.crosstab(df_expl['Label_List'], df_expl['Mood_Bin'])
        if not heatmap_data.empty:
            import seaborn as sns
            fig_hm, ax_hm = plt.subplots(figsize=(12, min(8, 0.5*len(heatmap_data))))
            sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd", ax=ax_hm, cbar=True)
            ax_hm.set_xlabel("Mood-Stufe")
            ax_hm.set_ylabel("Label")
            st.pyplot(fig_hm)
            st.caption("Heatmap: Zeigt, wie oft ein Label in verschiedenen Mood-Stufen vorkommt.")
        else:
            st.info("Keine Daten für Heatmap vorhanden.")
    else:
        st.info("Keine activities-Spalte in den Daten gefunden.")
else:
    st.info("Keine activities-Spalte in den Daten gefunden.")