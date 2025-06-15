import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import zscore
import seaborn as sns

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
- Identifikation von Tagen mit intra-täglichen Mixed States
- Aktivitäten-Filter, Marker und Kombinationsmuster
- Heatmap Aktivitäten vs. Mood
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

    # --- Aktivitäten vorbereiten ---
    if 'activities' in df.columns:
        df['Activities_List'] = df['activities'].fillna('').apply(lambda x: [a.strip() for a in x.split('|')] if x else [])
        all_activities = pd.Series([a for acts in df['Activities_List'] for a in acts if a])
        unique_activities = all_activities.value_counts().index.tolist()
    else:
        df['Activities_List'] = [[] for _ in range(len(df))]
        unique_activities = []

    # --- Mood-Score je Aktivität ---
    activity_mood = (
        df.explode('Activities_List')
        .groupby('Activities_List')['Stimmungswert']
        .mean()
        .drop('')
        .sort_values(ascending=False)
    )
    st.subheader("Durchschnittlicher Mood je Aktivität")
    st.bar_chart(activity_mood)

    # --- Filterfunktion: Activities (Sidebar) ---
    st.sidebar.header("Aktivitäten-Filter")
    selected_activities = st.sidebar.multiselect(
        "Nur Tage mit mind. einer dieser Aktivitäten zeigen:",
        unique_activities
    )
    if selected_activities:
        mask = df['Activities_List'].apply(lambda acts: any(a in acts for a in selected_activities))
        df_plot = df[mask].copy()
        st.info(f"Plots zeigen nur Tage mit mind. einer dieser Aktivitäten: {', '.join(selected_activities)}")
    else:
        df_plot = df.copy()

    # --- Warnsystem für kritische Aktivitäten ---
    critical_acts = ["meds not taken", "psychotic residual symptoms", "suicidal symptoms"]
    days_with_critical = df_plot['Activities_List'].apply(lambda acts: any(a in acts for a in critical_acts))
    if days_with_critical.any():
        st.error("🚨 **Kritische Aktivität:** Mindestens ein Tag mit riskanten Symptomen ('meds not taken', 'psychotic residual symptoms', 'suicidal symptoms')!")

    # --- Mood Zeitverlauf mit Markern für kritische Aktivitäten ---
    st.subheader("Mood Zeitverlauf (mit kritischen Aktivitäten-Markern)")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_plot['full_date'], df_plot['Stimmungswert'], color='gray', alpha=0.4, linewidth=1, label="Mood-Wert")
    ax.scatter(
        df_plot.loc[days_with_critical, 'full_date'],
        df_plot.loc[days_with_critical, 'Stimmungswert'],
        color='red', s=60, label='Kritische Aktivität', zorder=10, edgecolors='black'
    )
    ax.set_ylabel("Mood (1=Super Low ... 5=Super High)")
    ax.set_xlabel("Datum")
    ax.set_title("Mood Zeitverlauf mit kritischen Aktivitäten-Marker")
    ax.legend(loc='upper left')
    st.pyplot(fig)
    st.caption("""
        Interpretation: Rot markierte Punkte stehen für Tage mit kritischen Symptomen/fehlender Medikation. Sie bieten eine direkte Verbindung von Stimmung zu Risiko- oder Warn-Situationen.
    """)

    # --- Heatmap: Aktivitäten vs. Mood-Stufen ---
    st.subheader("Heatmap: Aktivitäten vs. Mood-Stufen")
    mood_bins = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    df_expl = df_plot.explode('Activities_List')
    df_expl = df_expl[df_expl['Activities_List'] != '']
    df_expl['Mood_Bin'] = pd.cut(df_expl['Stimmungswert'], bins=mood_bins, include_lowest=True, right=False)
    heatmap_data = pd.crosstab(df_expl['Activities_List'], df_expl['Mood_Bin'])
    fig_hm, ax_hm = plt.subplots(figsize=(12, min(8, 0.5*len(heatmap_data))))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd", ax=ax_hm, cbar=True)
    ax_hm.set_xlabel("Mood-Stufe")
    ax_hm.set_ylabel("Aktivität")
    st.pyplot(fig_hm)
    st.caption("""
        Interpretation: Diese Heatmap zeigt, wie häufig jede Aktivität bei verschiedenen Mood-Stufen eingetragen wurde. Dunklere Farben = häufiger. Auffällige Muster helfen bei der Suche nach möglichen Triggern oder Schutzfaktoren.
    """)

    # --- Kombinationsmuster (Top 10) ---
    st.subheader("Häufigste Aktivitäts-Kombinationen (Top 10)")
    comb_counts = pd.Series(
        [" | ".join(sorted(set(acts))) for acts in df_plot['Activities_List'] if len(acts) > 1]
    ).value_counts().head(10)
    st.dataframe(comb_counts.rename('Tage'), use_container_width=True)

    # Durchschnittlicher Mood je Kombi (Top 10)
    st.subheader("Durchschnittlicher Mood bei häufigen Aktivitäts-Kombinationen (Top 10)")
    comb_mood_dict = {}
    for combi, _ in comb_counts.items():
        acts = [a.strip() for a in combi.split('|')]
        mask = df_plot['Activities_List'].apply(lambda x: set(acts).issubset(set(x)) and len(x) == len(acts))
        comb_mood_dict[combi] = df_plot.loc[mask, 'Stimmungswert'].mean()
    comb_mood_df = pd.DataFrame(list(comb_mood_dict.items()), columns=['Aktivitäts-Kombination', 'Durchschnittlicher Mood']).sort_values('Durchschnittlicher Mood', ascending=False)
    st.dataframe(comb_mood_df, use_container_width=True)
    st.caption("""
        Interpretation: Diese Tabellen zeigen, welche Kombinationen von Aktivitäten am häufigsten auftreten und wie deine Stimmung an diesen Tagen im Schnitt war. So findest du günstige wie auch kritische Muster.
    """)

    # --- Mixed-State Grundlage: Tagesweise alle Mood-Listen sammeln ---
    daily_moods = df_plot.groupby('full_date')['Stimmungswert'].agg(list)
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

    # --- Tagesmittel für Plots ---
    df_tagesmittel = df_plot.groupby('full_date')['Stimmungswert'].mean().reset_index()
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
    loess_curve = lowess(raw, x, frac=loess_frac, return_sorted=False)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df_tagesmittel['Datum'], raw, color='gold', alpha=0.2, label="Tagesmittel (roh)")
    ax2.plot(df_tagesmittel['Datum'], sg, color='orange', linewidth=2, label="Savitzky-Golay")
    ax2.plot(df_tagesmittel['Datum'], loess_curve, color='crimson', linewidth=2, label="LOESS")
    for y in np.arange(1.5, 5.1, 0.5):
        ax2.axhline(y, color='lightgray', linestyle='--', linewidth=0.7)
    ax2.axhline(2.5, color='grey', linestyle='--', linewidth=1, alpha=0.6)
    ax2.text(df_tagesmittel['Datum'].iloc[5], 2.5 + 0.05, "Schwelle Depression (2.5)", color='grey', fontsize=9, va='bottom')
    ax2.axhline(3.5, color='grey', linestyle='--', linewidth=1, alpha=0.6)
    ax2.text(df_tagesmittel['Datum'].iloc[5], 3.5 + 0.05, "Schwelle Hypomanie (3.5)", color='grey', fontsize=9, va='bottom')
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

    # --- Frühwarnsignale: Varianz und Autokorrelation, z-score & Markierung bei gemeinsamer Spitze ---
    st.subheader("Überlagerte Frühwarnsignale: Varianz vs. Autokorrelation (z-score)")
    df_tagesmittel['Varianz_z'] = zscore(df_tagesmittel['Varianz'].fillna(0))
    df_tagesmittel['Autokorr_z'] = zscore(df_tagesmittel['Autokorr'].fillna(0))
    joint_peaks = (df_tagesmittel['Varianz_z'] > 1) & (df_tagesmittel['Autokorr_z'] > 1)
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df_tagesmittel['Datum'], df_tagesmittel['Varianz_z'], label='Varianz (z-score)')
    ax1.plot(df_tagesmittel['Datum'], df_tagesmittel['Autokorr_z'], label='Autokorrelation (z-score)')
    for i in range(1, len(joint_peaks)):
        if joint_peaks.iloc[i]:
            ax1.axvspan(df_tagesmittel['Datum'].iloc[i-1], df_tagesmittel['Datum'].iloc[i], color='red', alpha=0.2)
    ax1.set_title("Varianz & Autokorrelation (z-score, kritische Zeiträume rot)")
    ax1.set_xlabel("Datum")
    ax1.set_ylabel("z-score")
    ax1.legend()
    st.pyplot(fig1)

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

    # --- Statistik zu Mixed-States ---
    n_intra = int(df_intraday['Mixed_IntraDay'].sum())
    st.markdown(f"""
**Statistik Intra-tägliche Mixed-State-Tage:**  
- Intra-tägliche Mixed-States: **{n_intra}** von insgesamt **{len(df_intraday)}** Tagen
""")

else:
    st.info("Bitte lade zuerst eine Daylio-Export-CSV hoch.")