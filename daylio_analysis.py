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

# Zentrale Farbpalette (Thema: Warm, angelehnt an Heatmap)
COLORS = {
    "depressive_bg": "#D6EAF8", "euthym_bg": "#E8F8F5", "manic_bg": "#FEF9E7",
    "critical_marker": "#e41a1c", # Starkes Rot
    "main_line": "#ff7f00",      # Orange
    "secondary_line": "#e31a1c", # Ein anderes Rot
    "accent_1": "#f781bf",       # Pink/Lila
    "accent_2": "#a65628",       # Braun
    "grid": "#d3d3d3",
    "neutral_text": "#404040",
    "neutral_dots": "lightgray"
}

# --- KERNFUNKTIONEN (unverändert) ---
def get_threshold(metric: str, win: int, level: str) -> float:
    intercept, slope = THRESHOLDS[metric][level]
    if metric == 'varianz':
        return max(0.2 if level == 'warn' else 0.3, intercept + slope * win)
    return intercept + slope * win

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
    df_intraday = pd.DataFrame({
        'Datum': pd.to_datetime(daily_moods.index),
        'Mood_List': daily_moods.values,
        'Mixed_IntraDay': (daily_moods.apply(lambda x: max(x) - min(x) if len(x) > 1 else 0) >= 2) | (daily_moods.apply(np.std) >= 1)
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

# --- VISUALISIERUNGS-HELFER ---

def style_plot(ax: plt.Axes):
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color=COLORS['grid'], alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])

def add_mood_zones(ax: plt.Axes):
    ax.axhspan(1, 2.5, facecolor=COLORS['depressive_bg'], alpha=0.5, zorder=-1)
    ax.axhspan(2.5, 3.5, facecolor=COLORS['euthym_bg'], alpha=0.6, zorder=-1)
    ax.axhspan(3.5, 5, facecolor=COLORS['manic_bg'], alpha=0.5, zorder=-1)

def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

# --- ÜBERARBEITETE PLOT-FUNKTIONEN MIT NEUEM STIL & SICHTBARKEIT ---

def plot_histogram(df_hist: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(1, 5.6, 0.5)
    hist_values, _ = np.histogram(df_hist['Stimmungswert'].dropna(), bins=bins)
    
    bar_colors = plt.cm.YlOrRd(np.linspace(0.4, 0.8, len(bins)-1))
    ax.bar(bins[:-1] + 0.25, hist_values, width=0.45, color=bar_colors, edgecolor='white', linewidth=1)
    
    style_plot(ax)
    ax.set_title(title)
    ax.set_xlabel("Stimmungswert (Tagesmittel)")
    ax.set_ylabel("Anzahl Tage")
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "haeufigkeitsverteilung.png")
    with st.expander("Interpretation"):
        st.caption("""**Was es zeigt:** Die Häufigkeit, mit der bestimmte tägliche Stimmungsmittelwerte aufgetreten sind. **Warum es wichtig ist:** Sie erkennen auf einen Blick, ob Ihre Stimmung zu bestimmten Bereichen (z.B. depressiv, euthym) tendiert.""")

def plot_mood_timeseries(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    add_mood_zones(ax)
    
    # Farbverlauf basierend auf Stimmungswert
    cmap = plt.cm.get_cmap('YlOrRd')
    colors = cmap( (df['Stimmungswert'] - 1) / 4 )
    
    ax.plot(df['Datum'], df['Stimmungswert'], color='gray', alpha=0.3, linewidth=1, zorder=1)
    ax.scatter(df['Datum'], df['Stimmungswert'], c=colors, s=40, zorder=2, edgecolors='white', linewidth=0.5)
    
    style_plot(ax)
    ax.set_title("Stimmungsverlauf mit Phasen-Zonen")
    ax.set_ylabel("Stimmung (1-5)")
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "mood_zeitverlauf.png")
    with st.expander("Interpretation"):
        st.caption("""**Was es zeigt:** Den täglichen Verlauf Ihrer Stimmungsmittelwerte. **Warum es wichtig ist:** Macht Muster wie Phasen, abrupte Wechsel oder Zyklen sichtbar. Die Hintergrundfarbe zeigt die Stimmungszone an.""")

def plot_smoothing(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    add_mood_zones(ax)
    
    # Tägliche Mittelwerte als sichtbare Punkte im Hintergrund
    ax.scatter(df['Datum'], df['Stimmungswert'], color=COLORS['neutral_dots'], alpha=0.6, label="Tagesmittel (roh)", s=15, zorder=1)
    
    ax.plot(df['Datum'], df['SG_Smooth'], color=COLORS['secondary_line'], linewidth=2.5, label="Savitzky-Golay Trend", zorder=3)
    ax.plot(df['Datum'], df['LOESS_Smooth'], color=COLORS['main_line'], linewidth=2.5, label="LOESS Trend", zorder=4)
    
    style_plot(ax)
    ax.set_title("Geglättete Stimmungstrends")
    ax.set_ylabel("Geglätteter Stimmungswert")
    ax.legend(loc='upper left')
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "stimmungsglaettung.png")
    with st.expander("Interpretation"):
        st.caption("""**Was es zeigt:** Langfristige Stimmungstrends, die das tägliche "Rauschen" herausfiltern. **Warum es wichtig ist:** Offenbart langsame, unterschwellige Veränderungen wie den Beginn einer neuen Episode oder saisonale Muster.""")

def plot_early_warning_signals(df: pd.DataFrame, win: int):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(df['Datum'], df['Varianz'], color=COLORS['main_line'], label=f'Varianz ({win} Tage)')
    ax.fill_between(df['Datum'], df['Varianz'], color=COLORS['main_line'], alpha=0.2)
    ax.set_ylabel("Varianz", color=COLORS['main_line'])
    
    ax2 = ax.twinx()
    ax2.plot(df['Datum'], df['Autokorr'], color=COLORS['secondary_line'], label=f'Autokorrelation ({win} Tage)')
    ax2.fill_between(df['Datum'], df['Autokorr'], color=COLORS['secondary_line'], alpha=0.2)
    ax2.set_ylabel("Autokorrelation", color=COLORS['secondary_line'])
    
    style_plot(ax)
    ax.set_title("Frühwarnsignale: Varianz & Autokorrelation")
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(COLORS['grid'])
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "fruehwarnsignale.png")
    with st.expander("Interpretation"):
        st.caption("""**Was es zeigt:** Varianz (Schwankungsbreite) und Autokorrelation (Trägheit). **Warum es wichtig ist:** Ein plötzlicher Anstieg beider Werte kann einen bevorstehenden Phasenwechsel signalisieren.""")

def plot_entropy(df: pd.DataFrame, entropy_win: int):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(df['Datum'], df['Shannon Entropy'], color=COLORS['accent_2'], label=f'Shannon Entropie ({entropy_win} Tage)')
    ax.fill_between(df['Datum'], df['Shannon Entropy'], color=COLORS['accent_2'], alpha=0.2)
    ax.plot(df['Datum'], df['Approximate Entropy'], color=COLORS['accent_1'], label=f'Approximate Entropie ({entropy_win} Tage)')
    ax.fill_between(df['Datum'], df['Approximate Entropy'], color=COLORS['accent_1'], alpha=0.2)
    
    style_plot(ax)
    ax.set_title("Stabilitätsanalyse: Entropie")
    ax.set_ylabel("Entropie (Komplexität)")
    ax.legend(loc='upper left')
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "entropie.png")
    with st.expander("Interpretation"):
        st.caption("""**Was es zeigt:** Die Komplexität und Unvorhersagbarkeit Ihrer Stimmung. **Warum es wichtig ist:** Hohe Werte deuten auf chaotische, instabile Phasen hin. Niedrige Werte stehen für Stabilität.""")

def plot_mixed_states(df_intraday: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(15, 5))
    add_mood_zones(ax)
    
    daily_means = [np.mean(m) for m in df_intraday['Mood_List']]
    mixed_days_mask = df_intraday['Mixed_IntraDay']
    
    # Alle Tage als verbundene, graue Punkte
    ax.plot(df_intraday['Datum'], daily_means, color='gray', marker='.', linestyle='-', alpha=0.6, label='Tagesmittelwert')
    
    # Kritische Tage als große, rote Punkte darüber
    ax.scatter(df_intraday['Datum'][mixed_days_mask], np.array(daily_means)[mixed_days_mask], color=COLORS['critical_marker'], label='Hohe intra-tägliche Schwankung', s=80, zorder=5, edgecolors='black', linewidth=1)
    
    style_plot(ax)
    ax.set_title("Analyse intra-täglicher Stimmungsschwankungen")
    ax.set_ylabel('Stimmungswert')
    ax.legend(loc='upper left')
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "mixed_states.png")
    with st.expander("Interpretation"):
        st.caption("""**Was es zeigt:** Tage (rot markiert), an denen die Stimmung innerhalb eines Tages sehr stark geschwankt hat. **Warum es wichtig ist:** Dies deckt "Mixed States" oder hohe Instabilität auf, die im Tagesmittelwert untergehen würden.""")
        
def plot_label_heatmap(df_raw: pd.DataFrame):
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
    ax.set_title("Heatmap: Aktivitäten vs. Stimmung", loc='left', pad=20)
    ax.set_xlabel("Stimmungskategorie")
    ax.set_ylabel("Aktivität / Label")
    plt.tight_layout()
    st.pyplot(fig)
    st.download_button("Download Heatmap", fig_to_bytes(fig), "label_heatmap.png")
    with st.expander("Interpretation"):
        st.caption("""**Was es zeigt:** Wie oft eine Aktivität bei einer Stimmung auftrat. **Warum es wichtig ist:** Hilft, Trigger oder hilfreiche Bewältigungsstrategien zu identifizieren.""")
        
# --- STREAMLIT HAUPTANWENDUNG ---
def main():
    st.title("Daylio Stimmungsanalyse")
    st.write("Laden Sie Ihren Daylio-Export (CSV) hoch, um Visualisierungen und Frühwarnsignale für Ihre Stimmungsdynamik zu erhalten.")
    
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

        st.subheader("Häufigkeitsverteilung der Stimmung")
        filter_art = st.selectbox("Zeitfenster für Verteilung:", ["Gesamter Zeitraum", "Jahresweise", "Monatsweise"], key="dist_filter")
        df_hist = df_daily.copy()
        title = "Häufigkeitsverteilung (Gesamter Zeitraum)"
        if filter_art == "Jahresweise":
            jahre = sorted(df_daily['Datum'].dt.year.unique())
            jahr = st.selectbox("Jahr auswählen:", jahre, index=len(jahre)-1)
            df_hist = df_daily[df_daily['Datum'].dt.year == jahr]
            title = f"Häufigkeitsverteilung ({jahr})"
        elif filter_art == "Monatsweise":
            jahre = sorted(df_daily['Datum'].dt.year.unique())
            jahr = st.selectbox("Jahr auswählen:", jahre, index=len(jahre)-1)
            monate = sorted(df_daily[df_daily['Datum'].dt.year == jahr]['Datum'].dt.month.unique())
            monat = st.selectbox("Monat auswählen:", monate, index=len(monate)-1, format_func=lambda m: f"{m:02d}")
            df_hist = df_daily[(df_daily['Datum'].dt.year == jahr) & (df_daily['Datum'].dt.month == monat)]
            title = f"Häufigkeitsverteilung ({jahr}-{monat:02d})"
        plot_histogram(df_hist, title)
        
        st.subheader("Tagesmittelwerte der Stimmung im Zeitverlauf")
        plot_mood_timeseries(df_daily)
        
        st.subheader("Geglättete Stimmungstrends")
        plot_smoothing(df_daily)
        
        st.subheader("Frühwarnsignale & Stabilität")
        plot_early_warning_signals(df_daily, win)
        plot_entropy(df_daily, entropy_win)
        
        st.subheader("Analyse intra-täglicher Stimmungsschwankungen")
        plot_mixed_states(df_intraday)
        n_intra = int(df_intraday['Mixed_IntraDay'].sum())
        st.metric(label="Tage mit hoher intra-täglicher Varianz", value=f"{n_intra}", delta=f"{n_intra/len(df_intraday):.1%} der Tage")
        
        st.subheader("Analyse der Aktivitäten-Label")
        plot_label_heatmap(df_raw)

    else:
        st.info("Bitte laden Sie eine Daylio-Export-CSV-Datei hoch, um die Analyse zu starten.")

if __name__ == '__main__':
    main()