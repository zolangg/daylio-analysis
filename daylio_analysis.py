import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
import io
from typing import Tuple, Dict, Any, List

# --- KONSTANTEN UND KONFIGURATION ---
st.set_page_config(layout="wide", page_title="Daylio Stimmungsanalyse")

MOOD_MAP = {
    'Super Low': 1, 'Low': 2, 'Euthym': 3, 'High': 4, 'Super High': 5
}

THRESHOLDS = {
    'varianz': {'warn': (0.4, -0.0015), 'kritisch': (0.5, -0.001)},
    'autokorr': {'warn': (0.45, -0.0005), 'kritisch': (0.65, -0.0003)},
    'shannon': {'warn': (1.3, 0.001), 'kritisch': (1.6, 0.001)},
    'apen': {'warn': (0.5, 0.001), 'kritisch': (0.7, 0.0015)},
}

def get_threshold(metric: str, win: int, level: str) -> float:
    """Berechnet einen dynamischen Schwellenwert basierend auf der Fenstergröße."""
    intercept, slope = THRESHOLDS[metric][level]
    if metric == 'varianz':
        return max(0.2 if level == 'warn' else 0.3, intercept + slope * win)
    return intercept + slope * win

# --- KERNFUNKTIONEN (BERECHNUNG) ---

def shannon_entropy(sequence: np.ndarray) -> float:
    _, counts = np.unique(sequence, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))

def approximate_entropy(U: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    U = np.asarray(U)
    N = len(U)
    r = r_factor * np.std(U)
    if r < 1e-6: return 0.0
    if N <= m + 1: return np.nan
    def _phi(m_val: int) -> float:
        x = np.array([U[i:i + m_val] for i in range(N - m_val + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) / (N - m_val + 1.0)
        return np.sum(np.log(C + 1e-12)) / (N - m_val + 1.0)
    return abs(_phi(m) - _phi(m + 1))

# --- DATENVERARBEITUNG (mit Caching für Performance) ---

@st.cache_data
def load_and_preprocess_data(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df['full_date'] = pd.to_datetime(df['full_date'])
    df = df.sort_values('full_date')
    df['Stimmungswert'] = df['mood'].map(MOOD_MAP)
    return df

@st.cache_data
def calculate_metrics(df: pd.DataFrame, win: int, entropy_win: int, loess_frac: float, sg_win: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily_moods = df.groupby(df['full_date'].dt.date)['Stimmungswert'].agg(list)
    intra_var = daily_moods.apply(np.std)
    intra_range = daily_moods.apply(lambda x: max(x) - min(x) if len(x) > 1 else 0)
    df_intraday = pd.DataFrame({
        'Datum': pd.to_datetime(daily_moods.index),
        'Mood_List': daily_moods.values,
        'Mixed_IntraDay': (intra_range >= 2) | (intra_var >= 1)
    })
    df_daily = df.groupby(df['full_date'].dt.date)['Stimmungswert'].mean().reset_index()
    df_daily = df_daily.rename(columns={'full_date': 'Datum'})
    df_daily['Datum'] = pd.to_datetime(df_daily['Datum'])
    df_daily = df_daily.sort_values('Datum')
    df_daily['Varianz'] = df_daily['Stimmungswert'].rolling(window=win, min_periods=win//2).var()
    df_daily['Autokorr'] = df_daily['Stimmungswert'].rolling(window=win, min_periods=win//2).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
    df_daily['Shannon Entropy'] = df_daily['Stimmungswert'].rolling(window=entropy_win, min_periods=entropy_win//2).apply(shannon_entropy, raw=True)
    df_daily['Approximate Entropy'] = df_daily['Stimmungswert'].rolling(window=entropy_win, min_periods=entropy_win//2).apply(approximate_entropy, raw=True)
    raw = df_daily['Stimmungswert'].values
    if len(raw) >= sg_win:
        df_daily['SG_Smooth'] = savgol_filter(raw, window_length=sg_win, polyorder=3)
    else:
        df_daily['SG_Smooth'] = np.nan
    df_daily['LOESS_Smooth'] = lowess(raw, np.arange(len(raw)), frac=loess_frac, return_sorted=False)
    return df_daily, df_intraday

# --- VISUALISIERUNGS-FUNKTIONEN ---

def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

def plot_histogram(df_hist: pd.DataFrame):
    """Erstellt und zeigt ein Histogramm der Stimmungsverteilung."""
    bins = np.arange(1, 5.6, 0.5)
    hist_values, _ = np.histogram(df_hist['Stimmungswert'].dropna(), bins=bins)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bins[:-1] + 0.25, hist_values, width=0.5, color='skyblue', edgecolor='black')
    ax.set_xticks(bins[:-1] + 0.25)
    ax.set_xticklabels([f"{b:.1f}" for b in bins[:-1]])
    ax.set_xlabel("Stimmungswert (Tagesmittel)")
    ax.set_ylabel("Anzahl Tage")
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "haeufigkeitsverteilung.png")
    with st.expander("Interpretation"):
        st.caption("""
        **Was es zeigt:** Die Häufigkeit, mit der bestimmte tägliche Stimmungsmittelwerte aufgetreten sind.
        
        **Warum es wichtig ist:** Sie erkennen auf einen Blick, ob Ihre Stimmung zu bestimmten Bereichen (z.B. depressiv, euthym) tendiert und ob es mehrere "Gipfel" gibt, was auf eine bipolare Verteilung hindeuten könnte.
        """)

def plot_mood_timeseries(df: pd.DataFrame):
    """Erstellt und zeigt den farbcodierten Zeitverlauf der Stimmung."""
    mood_vals = df['Stimmungswert'].values
    dates = df['Datum'].values
    conditions = [mood_vals <= 1.5, mood_vals <= 2.0, mood_vals <= 2.5, mood_vals <= 3.0, mood_vals <= 3.5, mood_vals <= 4.0, mood_vals <= 4.5]
    colors = ['#8e44ad', '#5e72b0', '#3498db', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
    color_array = np.select(conditions, colors, default='#900c3f')
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(dates, mood_vals, c=color_array, s=30, label="Tagesmittelwert")
    ax.plot(dates, mood_vals, color='gray', alpha=0.4, linewidth=1, zorder=-1)
    for y in np.arange(1.5, 5.1, 0.5):
        ax.axhline(y, color='lightgray', linestyle='--', linewidth=0.5)
    ax.set_ylabel("Stimmung (1=Super Low ... 5=Super High)")
    ax.set_xlabel("Datum")
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "mood_zeitverlauf.png")
    with st.expander("Interpretation"):
        st.caption("""
        **Was es zeigt:** Den täglichen Verlauf Ihrer Stimmungsmittelwerte über die Zeit.
        
        **Warum es wichtig ist:** Dieser Plot macht Muster wie Phasen, abrupte Wechsel oder Zyklen direkt sichtbar. Die Farbcodierung hilft, Stimmungsbereiche schnell zu identifizieren.
        """)

def plot_smoothing(df: pd.DataFrame):
    """Erstellt und zeigt die geglätteten Stimmungskurven."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Datum'], df['Stimmungswert'], color='gold', alpha=0.2, label="Tagesmittel (roh)")
    ax.plot(df['Datum'], df['SG_Smooth'], color='orange', linewidth=2, label="Savitzky-Golay")
    ax.plot(df['Datum'], df['LOESS_Smooth'], color='crimson', linewidth=2, label="LOESS")
    for y_val, label in [(2.5, "Schwelle Depression"), (3.5, "Schwelle Hypomanie")]:
        ax.axhline(y_val, color='grey', linestyle='--', linewidth=1, alpha=0.6)
        ax.text(df['Datum'].iloc[5], y_val + 0.05, label, color='grey', fontsize=9, va='bottom')
    ax.set_xlabel("Datum")
    ax.set_ylabel("Geglätteter Stimmungswert")
    ax.legend(loc='upper left')
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "stimmungsglaettung.png")
    with st.expander("Interpretation"):
        st.caption("""
        **Was es zeigt:** Langfristige Stimmungstrends durch mathematische Glättung (Savitzky-Golay, LOESS), die das tägliche "Rauschen" herausfiltert.
        
        **Warum es wichtig ist:** Die Glättung offenbart langsame, unterschwellige Veränderungen wie den Beginn einer neuen Episode oder saisonale Muster, die im Tagesverlauf untergehen können.
        """)
        
def plot_early_warning_signals(df: pd.DataFrame, win: int):
    """Erstellt den Plot für Varianz und Autokorrelation."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Datum'], df['Varianz'], color='gold', label=f'Rollierende Varianz ({win} Tage)')
    ax.plot(df['Datum'], df['Autokorr'], color='orange', label=f'Roll. Autokorrelation ({win} Tage)')
    for metric in ['varianz', 'autokorr']:
        ax.axhline(get_threshold(metric, win, 'warn'), color='grey', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(get_threshold(metric, win, 'kritisch'), color='grey', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Wert")
    ax.legend(loc='upper left')
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "fruehwarnsignale.png")
    with st.expander("Interpretation"):
        st.caption("""
        **Was es zeigt:** Zwei Frühwarnindikatoren im Zeitverlauf.
        - **Varianz:** Die Stärke der Stimmungsschwankungen.
        - **Autokorrelation:** Die "Trägheit" der Stimmung (wie sehr der heutige Wert dem gestrigen ähnelt).
        
        **Warum es wichtig ist:** Ein plötzlicher Anstieg beider Werte kann einen bevorstehenden Phasenwechsel (z.B. in eine Depression oder Manie) signalisieren, noch bevor dieser vollständig eintritt.
        """)

def plot_entropy(df: pd.DataFrame, entropy_win: int):
    """Erstellt den Plot für die Entropie-Maße."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Datum'], df['Shannon Entropy'], label=f'Shannon Entropie ({entropy_win} Tage)', color='blue')
    ax.plot(df['Datum'], df['Approximate Entropy'], label=f'Approximate Entropie ({entropy_win} Tage)', color='red')
    for metric in ['shannon', 'apen']:
        ax.axhline(get_threshold(metric, entropy_win, 'warn'), color='gray', linestyle='--', alpha=0.5)
        ax.axhline(get_threshold(metric, entropy_win, 'kritisch'), color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Entropie (Komplexität)")
    ax.legend(loc='upper left')
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "entropie.png")
    with st.expander("Interpretation"):
        st.caption("""
        **Was es zeigt:** Die Komplexität und Unvorhersagbarkeit (Entropie) Ihrer Stimmung.
        
        **Warum es wichtig ist:** Hohe Entropiewerte deuten auf chaotische, instabile Phasen hin (z.B. "Mixed States"). Niedrige Werte stehen für sehr stabile, gleichförmige Phasen. Ein Anstieg kann eine Destabilisierung ankündigen.
        """)
        
def plot_mixed_states(df_intraday: pd.DataFrame):
    """Erstellt den Plot für intra-tägliche gemischte Zustände."""
    fig, ax = plt.subplots(figsize=(15, 5))
    daily_means = [np.mean(m) for m in df_intraday['Mood_List']]
    mixed_days_mask = df_intraday['Mixed_IntraDay']
    ax.plot(df_intraday['Datum'], daily_means, color='lightgray', label='Tagesmittelwert', marker='.', linestyle='-')
    ax.scatter(df_intraday['Datum'][mixed_days_mask], np.array(daily_means)[mixed_days_mask], color='crimson', label='Hohe intra-tägliche Schwankung', s=50, zorder=5)
    ax.set_ylabel('Stimmungswert')
    ax.set_xlabel("Datum")
    ax.legend(loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "mixed_states.png")
    with st.expander("Interpretation"):
        st.caption("""
        **Was es zeigt:** Tage (rot markiert), an denen die Stimmung *innerhalb eines Tages* sehr stark geschwankt hat (z.B. von "Low" zu "High").
        
        **Warum es wichtig ist:** Dies deckt "Mixed States" oder hohe Instabilität auf Tagesebene auf, die im reinen Tagesmittelwert untergehen würden. Häufige Markierungen sind ein starkes Zeichen für emotionale Dysregulation.
        """)

def plot_label_heatmap(df_raw: pd.DataFrame):
    """Erstellt und zeigt eine Heatmap der Label-Verteilung über die Stimmung."""
    if 'activities' not in df_raw.columns or df_raw['activities'].isnull().all():
        st.info("Keine 'activities'-Spalte für die Label-Analyse gefunden.")
        return
    df_raw['Label_List'] = df_raw['activities'].fillna('').str.split('|').apply(lambda x: [label.strip() for label in x if label.strip()])
    df_expl = df_raw.explode('Label_List')
    if df_expl.empty or df_expl['Label_List'].nunique() == 0:
        st.info("Keine Labels zur Analyse vorhanden.")
        return
    mood_bins = [1, 2, 3, 4, 5, 6]
    mood_labels = ['Super Low (1)', 'Low (2)', 'Euthym (3)', 'High (4)', 'Super High (5)']
    df_expl['Mood_Bin'] = pd.cut(df_expl['Stimmungswert'], bins=mood_bins, labels=mood_labels, right=False, include_lowest=True)
    heatmap_data = pd.crosstab(df_expl['Label_List'], df_expl['Mood_Bin'])
    if heatmap_data.empty: return
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(heatmap_data))))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar=True, linewidths=.5)
    ax.set_xlabel("Stimmungskategorie")
    ax.set_ylabel("Aktivität / Label")
    plt.tight_layout()
    st.pyplot(fig)
    st.download_button("Download Heatmap", fig_to_bytes(fig), "label_heatmap.png")
    with st.expander("Interpretation"):
        st.caption("""
        **Was es zeigt:** Eine Heatmap, die zählt, wie oft eine bestimmte Aktivität (Label) bei einer bestimmten Stimmungskategorie protokolliert wurde.
        
        **Warum es wichtig ist:** Sie können Muster erkennen, welche Aktivitäten mit positiven oder negativen Stimmungen korrelieren. Dies hilft, Trigger oder hilfreiche Bewältigungsstrategien zu identifizieren.
        """)

# --- STREAMLIT HAUPTANWENDUNG ---
def main():
    st.title("Daylio Stimmungsanalyse")
    st.write("""
    Laden Sie Ihren Daylio-Export (CSV) hoch, um Visualisierungen und Frühwarnsignale für Ihre Stimmungsdynamik zu erhalten.
    Die Analyse umfasst Verteilungen, Zeitverläufe, Stabilitätsmaße (Varianz, Entropie) und eine Label-Analyse.
    """)
    uploaded_file = st.file_uploader("Daylio CSV-Datei hochladen", type=["csv"])
    if uploaded_file:
        st.sidebar.header("Einstellungen für die Analyse")
        win = st.sidebar.slider("Roll. Fenster (Varianz/Autokorr) [Tage]", 7, 180, 90)
        entropy_win = st.sidebar.slider("Entropie-Fenster [Tage]", 7, 180, 60)
        sg_win = st.sidebar.slider("Savitzky-Golay Glättung [Tage]", 7, 61, 31, step=2)
        loess_frac = st.sidebar.slider("LOESS Glättung (Anteil Datenpunkte)", 0.05, 0.25, 0.08)

        df_raw = load_and_preprocess_data(uploaded_file)
        df_daily, df_intraday = calculate_metrics(df_raw, win, entropy_win, loess_frac, sg_win)

        warnings = []
        last = df_daily.iloc[-1]
        if not pd.isna(last['Varianz']) and last['Varianz'] > get_threshold('varianz', win, 'kritisch'): warnings.append(f"**Varianz kritisch:** {last['Varianz']:.2f}")
        if not pd.isna(last['Autokorr']) and last['Autokorr'] > get_threshold('autokorr', win, 'kritisch'): warnings.append(f"**Autokorrelation kritisch:** {last['Autokorr']:.2f}")
        if not pd.isna(last['Shannon Entropy']) and last['Shannon Entropy'] > get_threshold('shannon', entropy_win, 'kritisch'): warnings.append(f"**Shannon Entropie kritisch:** {last['Shannon Entropy']:.2f}")
        if not pd.isna(last['Approximate Entropy']) and last['Approximate Entropy'] > get_threshold('apen', entropy_win, 'kritisch'): warnings.append(f"**Approximate Entropy kritisch:** {last['Approximate Entropy']:.2f}")
        if warnings: st.error("🚨 **KRITISCHE WARNUNG (basierend auf dem letzten Wert):**\n\n" + "\n\n".join(warnings))

        # --- Dashboard mit Plots erstellen (NEUE REIHENFOLGE) ---
        st.subheader(f"Häufigkeitsverteilung der Stimmung")
        filter_art = st.selectbox("Zeitfenster für Verteilung:", ["Gesamter Zeitraum", "Jahresweise", "Monatsweise"], key="dist_filter")
        df_hist, jahr, monat = df_daily.copy(), None, None
        if filter_art != "Gesamter Zeitraum":
            jahre = sorted(df_daily['Datum'].dt.year.unique())
            jahr = st.selectbox("Jahr auswählen:", jahre, index=len(jahre)-1)
            if filter_art == "Monatsweise":
                monate = sorted(df_daily[df_daily['Datum'].dt.year == jahr]['Datum'].dt.month.unique())
                monat = st.selectbox("Monat auswählen:", monate, index=len(monate)-1, format_func=lambda m: f"{m:02d}")
                df_hist = df_daily[(df_daily['Datum'].dt.year == jahr) & (df_daily['Datum'].dt.month == monat)]
            else:
                df_hist = df_daily[df_daily['Datum'].dt.year == jahr]
        plot_histogram(df_hist)
        
        st.subheader("Tagesmittelwerte der Stimmung im Zeitverlauf")
        plot_mood_timeseries(df_daily)
        
        st.subheader("Geglättete Stimmungstrends (LOESS & Savitzky-Golay)")
        plot_smoothing(df_daily)
        
        st.subheader("Frühwarnsignale: Rollierende Varianz & Autokorrelation")
        plot_early_warning_signals(df_daily, win)
        
        st.subheader("Stabilitätsanalyse: Shannon- & Approximate-Entropie")
        plot_entropy(df_daily, entropy_win)
        
        st.subheader("Analyse intra-täglicher Stimmungsschwankungen (Mixed States)")
        plot_mixed_states(df_intraday)
        n_intra = int(df_intraday['Mixed_IntraDay'].sum())
        st.metric(label="Tage mit hoher intra-täglicher Varianz", value=f"{n_intra}", delta=f"{n_intra/len(df_intraday):.1%} der Tage")
        
        st.subheader("Analyse der Aktivitäten-Label (Heatmap)")
        plot_label_heatmap(df_raw)

    else:
        st.info("Bitte laden Sie eine Daylio-Export-CSV-Datei hoch, um die Analyse zu starten.")

if __name__ == '__main__':
    main()