import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter, find_peaks
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

# --- VISUALISIERUNGS-FUNKTIONEN (inkl. neuer Zyklus-Plot) ---

def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

# (Bestehende Plot-Funktionen bleiben unverändert)
def plot_histogram(df_hist: pd.DataFrame):
    bins = np.arange(1, 5.6, 0.5)
    hist_values, _ = np.histogram(df_hist['Stimmungswert'].dropna(), bins=bins)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bins[:-1] + 0.25, hist_values, width=0.5, color='skyblue', edgecolor='black')
    ax.set_xticks(bins[:-1] + 0.25)
    ax.set_xticklabels([f"{b:.1f}" for b in bins[:-1]])
    ax.set_xlabel("Stimmungswert (Tagesmittel)")
    ax.set_ylabel("Anzahl Tage")
    st.pyplot(fig)

def plot_mood_timeseries(df: pd.DataFrame):
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

def plot_smoothing(df: pd.DataFrame):
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
        
def plot_early_warning_signals(df: pd.DataFrame, win: int):
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

def plot_entropy(df: pd.DataFrame, entropy_win: int):
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
        
def plot_mixed_states(df_intraday: pd.DataFrame):
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

# --- NEUE FUNKTIONEN FÜR ZYKLUSANALYSE ---

@st.cache_data
def calculate_cycle_lengths(data_series: pd.Series, prominence: float, distance: int) -> tuple:
    """Findet Hoch- und Tiefpunkte und berechnet die Abstände."""
    if data_series.isnull().all():
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    # Hochpunkte (im hypomanischen Bereich > 3.5)
    peaks, _ = find_peaks(data_series, height=3.5, prominence=prominence, distance=distance)
    # Tiefpunkte (im depressiven Bereich < 2.5)
    troughs, _ = find_peaks(-data_series, height=-2.5, prominence=prominence, distance=distance)

    peak_diffs = np.diff(peaks)
    trough_diffs = np.diff(troughs)
    
    return peaks, troughs, peak_diffs, trough_diffs

def plot_cycle_analysis(df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray):
    """Visualisiert die geglättete Kurve mit markierten Hoch- und Tiefpunkten."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Geglättete Kurve plotten
    ax.plot(df['Datum'], df['LOESS_Smooth'], color='gray', linewidth=1.5, label="LOESS Trend")
    
    # Hintergrundzonen für Klarheit
    ax.axhspan(1, 2.5, facecolor='lightblue', alpha=0.3, label='Depressive Zone')
    ax.axhspan(3.5, 5, facecolor='lightyellow', alpha=0.4, label='(Hypo)Manische Zone')
    
    # Markierungen für Peaks und Troughs
    if len(peaks) > 0:
        ax.plot(df['Datum'].iloc[peaks], df['LOESS_Smooth'].iloc[peaks], "v", color='red', markersize=10, label='Hochpunkt')
    if len(troughs) > 0:
        ax.plot(df['Datum'].iloc[troughs], df['LOESS_Smooth'].iloc[troughs], "^", color='blue', markersize=10, label='Tiefpunkt')
    
    ax.set_xlabel("Datum")
    ax.set_ylabel("Geglätteter Stimmungswert")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "zyklusanalyse.png")
    with st.expander("Interpretation"):
        st.caption("""
        **Was es zeigt:** Die geglättete Stimmungskurve mit automatisch erkannten Hoch- (rot) und Tiefpunkten (blau). Die Analyse misst den Abstand in Tagen zwischen diesen Punkten.
        
        **Warum es wichtig ist:** Diese Analyse hilft, die Regelmäßigkeit und Dauer Ihrer Stimmungszyklen zu objektivieren. Sie sehen, ob es ein wiederkehrendes Muster gibt (z.B. "alle 30-40 Tage ein Tiefpunkt"), was für die Diagnostik (z.B. Rapid Cycling) und das persönliche Management relevant sein kann.
        """)
        
# --- STREAMLIT HAUPTANWENDUNG ---
def main():
    st.title("Daylio Stimmungsanalyse")
    st.write("Laden Sie Ihren Daylio-Export (CSV) hoch, um Visualisierungen, Frühwarnsignale und eine Zyklusanalyse für Ihre Stimmungsdynamik zu erhalten.")
    
    uploaded_file = st.file_uploader("Daylio CSV-Datei hochladen", type=["csv"])
    if uploaded_file:
        # --- Sidebar für Einstellungen ---
        st.sidebar.header("Allgemeine Einstellungen")
        loess_frac = st.sidebar.slider("LOESS Glättung (Anteil Datenpunkte)", 0.05, 0.25, 0.1)
        
        st.sidebar.header("Frühwarnsignal-Einstellungen")
        win = st.sidebar.slider("Roll. Fenster (Varianz/Autokorr) [Tage]", 7, 180, 90)
        entropy_win = st.sidebar.slider("Entropie-Fenster [Tage]", 7, 180, 60)
        
        # NEUE EINSTELLUNGEN FÜR ZYKLUSANALYSE
        st.sidebar.header("Zyklusanalyse-Einstellungen")
        cycle_prominence = st.sidebar.slider("Prominenz (Signifikanz der Peaks)", 0.1, 2.0, 0.3, 0.1)
        cycle_distance = st.sidebar.slider("Minimaler Abstand der Peaks (Tage)", 7, 90, 21)

        # --- Berechnungen ---
        df_raw = load_and_preprocess_data(uploaded_file)
        # sg_win wird nicht mehr verwendet, daher entfernt
        df_daily, df_intraday = calculate_metrics(df_raw, win, entropy_win, loess_frac, sg_win=31) # sg_win ist hartcodiert, da nicht mehr im UI
        
        peaks, troughs, peak_diffs, trough_diffs = calculate_cycle_lengths(df_daily['LOESS_Smooth'], cycle_prominence, cycle_distance)

        # (Warnungs-Logik bleibt gleich)
        warnings = []
        last = df_daily.iloc[-1]
        if not pd.isna(last['Varianz']) and last['Varianz'] > get_threshold('varianz', win, 'kritisch'): warnings.append(f"**Varianz kritisch:** {last['Varianz']:.2f}")
        if not pd.isna(last['Autokorr']) and last['Autokorr'] > get_threshold('autokorr', win, 'kritisch'): warnings.append(f"**Autokorrelation kritisch:** {last['Autokorr']:.2f}")
        if not pd.isna(last['Shannon Entropy']) and last['Shannon Entropy'] > get_threshold('shannon', entropy_win, 'kritisch'): warnings.append(f"**Shannon Entropie kritisch:** {last['Shannon Entropy']:.2f}")
        if not pd.isna(last['Approximate Entropy']) and last['Approximate Entropy'] > get_threshold('apen', entropy_win, 'kritisch'): warnings.append(f"**Approximate Entropy kritisch:** {last['Approximate Entropy']:.2f}")
        if warnings: st.error("🚨 **KRITISCHE WARNUNG (basierend auf dem letzten Wert):**\n\n" + "\n\n".join(warnings))

        # --- Dashboard mit Plots ---
        st.subheader("Häufigkeitsverteilung der Stimmung")
        plot_histogram(df_daily)
        
        st.subheader("Tagesmittelwerte der Stimmung im Zeitverlauf")
        plot_mood_timeseries(df_daily)
        
        st.subheader("Geglättete Stimmungstrends (LOESS)")
        plot_smoothing(df_daily)
        
        # --- NEUER ABSCHNITT: ZYKLUSANALYSE ---
        st.subheader("Analyse der Stimmungszyklen")
        plot_cycle_analysis(df_daily, peaks, troughs)
        
        col1, col2 = st.columns(2)
        with col1:
            avg_peak_cycle = np.mean(peak_diffs) if len(peak_diffs) > 0 else "N/A"
            st.metric(label="Ø Zykluslänge (Hoch zu Hoch)", value=f"{avg_peak_cycle:.1f} Tage" if isinstance(avg_peak_cycle, float) else "Keine Zyklen gefunden")
        with col2:
            avg_trough_cycle = np.mean(trough_diffs) if len(trough_diffs) > 0 else "N/A"
            st.metric(label="Ø Zykluslänge (Tief zu Tief)", value=f"{avg_trough_cycle:.1f} Tage" if isinstance(avg_trough_cycle, float) else "Keine Zyklen gefunden")

        st.subheader("Frühwarnsignale & Stabilität")
        plot_early_warning_signals(df_daily, win)
        plot_entropy(df_daily, entropy_win)
        
        st.subheader("Analyse intra-täglicher Stimmungsschwankungen")
        plot_mixed_states(df_intraday)
        n_intra = int(df_intraday['Mixed_IntraDay'].sum())
        st.metric(label="Tage mit hoher intra-täglicher Varianz", value=f"{n_intra}", delta=f"{n_intra/len(df_intraday):.1%} der Tage")
        
    else:
        st.info("Bitte laden Sie eine Daylio-Export-CSV-Datei hoch, um die Analyse zu starten.")

if __name__ == '__main__':
    main()