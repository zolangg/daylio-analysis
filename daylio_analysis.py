import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
import io
from typing import Tuple

# --- KONSTANTEN UND KONFIGURATION ---
st.set_page_config(layout="wide", page_title="Daylio Stimmungsanalyse")

MOOD_MAP = {
    'Super Low': 1, 'Low': 2, 'Euthym': 3, 'High': 4, 'Super High': 5
}

# --- PLOT HELFERFUNKTIONEN ---

def style_plot(ax, title):
    """Einheitlicher Stil für alle Plots: Gitter, Spines entfernen, Titel."""
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.set_title(title, fontsize=14, pad=20)
    
def add_mood_zones(ax):
    """Fügt farbige Hintergrundzonen für Stimmungslevel hinzu."""
    ax.axhspan(1, 2.5, facecolor='#d6eaf8', alpha=0.4, zorder=-1, label='Depressiv')
    ax.axhspan(2.5, 3.5, facecolor='#e8f8f5', alpha=0.5, zorder=-1, label='Euthym')
    ax.axhspan(3.5, 5, facecolor='#fef9e7', alpha=0.4, zorder=-1, label='(Hypo)Manisch')

# (Alle Berechnungsfunktionen wie 'shannon_entropy', 'load_and_preprocess_data' etc. bleiben unverändert)
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['full_date'] = pd.to_datetime(df['full_date'])
    df = df.sort_values('full_date')
    df['Stimmungswert'] = df['mood'].map(MOOD_MAP)
    return df

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

def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf
# --- ÜBERARBEITETE VISUALISIERUNGS-FUNKTIONEN ---

def plot_histogram(df_hist: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(1, 5.6, 0.5)
    hist_values, _ = np.histogram(df_hist['Stimmungswert'].dropna(), bins=bins)
    
    ax.bar(bins[:-1] + 0.25, hist_values, width=0.4, color='#2ab7a9', edgecolor='white', linewidth=1.5)
    style_plot(ax, "Häufigkeitsverteilung der Tagesstimmung")
    ax.set_xlabel("Stimmungswert (Tagesmittel)")
    ax.set_ylabel("Anzahl Tage")
    
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "haeufigkeitsverteilung.png")

def plot_mood_timeseries(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    add_mood_zones(ax)
    
    ax.plot(df['Datum'], df['Stimmungswert'], color='gray', alpha=0.5, linewidth=1, zorder=1, label='Tagesmittel')
    ax.scatter(df['Datum'], df['Stimmungswert'], c=df['Stimmungswert'], cmap='viridis', s=30, zorder=2, edgecolors='white', linewidth=0.5)
    
    style_plot(ax, "Stimmungsverlauf mit Phasen-Zonen")
    ax.set_ylabel("Stimmung (1-5)")
    ax.legend(loc='upper left')
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "mood_zeitverlauf.png")

def plot_smoothing(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    add_mood_zones(ax)
    
    ax.plot(df['Datum'], df['Stimmungswert'], color='lightgray', alpha=0.4, label="Tagesmittel (roh)", marker='.', linestyle='None')
    ax.plot(df['Datum'], df['SG_Smooth'], color='#ff7f0e', linewidth=2.5, label="Savitzky-Golay Trend")
    ax.plot(df['Datum'], df['LOESS_Smooth'], color='#1f77b4', linewidth=2.5, label="LOESS Trend")
    
    style_plot(ax, "Geglättete Stimmungstrends")
    ax.set_ylabel("Geglätteter Stimmungswert")
    ax.legend(loc='upper left')
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "stimmungsglaettung.png")

def plot_early_warning_signals(df: pd.DataFrame, win: int):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Varianz
    ax.plot(df['Datum'], df['Varianz'], color='#ff7f0e', label=f'Varianz ({win} Tage)')
    ax.fill_between(df['Datum'], df['Varianz'], color='#ff7f0e', alpha=0.2)
    
    # Autokorrelation auf einer zweiten Y-Achse für bessere Skalierung
    ax2 = ax.twinx()
    ax2.plot(df['Datum'], df['Autokorr'], color='#1f77b4', label=f'Autokorrelation ({win} Tage)')
    ax2.fill_between(df['Datum'], df['Autokorr'], color='#1f77b4', alpha=0.2)
    
    style_plot(ax, "Frühwarnsignale: Varianz & Autokorrelation")
    ax.set_ylabel("Varianz", color='#ff7f0e')
    ax2.set_ylabel("Autokorrelation", color='#1f77b4')
    ax2.spines['right'].set_visible(True) # Rechte Achse sichtbar machen
    
    # Legenden kombinieren
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "fruehwarnsignale.png")

def plot_entropy(df: pd.DataFrame, entropy_win: int):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(df['Datum'], df['Shannon Entropy'], color='#d62728', label=f'Shannon Entropie ({entropy_win} Tage)')
    ax.fill_between(df['Datum'], df['Shannon Entropy'], color='#d62728', alpha=0.2)
    
    ax.plot(df['Datum'], df['Approximate Entropy'], color='#9467bd', label=f'Approximate Entropie ({entropy_win} Tage)')
    ax.fill_between(df['Datum'], df['Approximate Entropy'], color='#9467bd', alpha=0.2)
    
    style_plot(ax, "Entropie (Komplexität & Vorhersagbarkeit)")
    ax.set_ylabel("Entropie")
    ax.legend(loc='upper left')
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "entropie.png")

def plot_mixed_states(df_intraday: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(15, 5))
    add_mood_zones(ax)

    daily_means = [np.mean(m) for m in df_intraday['Mood_List']]
    mixed_days_mask = df_intraday['Mixed_IntraDay']
    
    ax.plot(df_intraday['Datum'], daily_means, color='gray', label='Tagesmittelwert', marker='.', linestyle='--', alpha=0.6)
    ax.scatter(
        df_intraday['Datum'][mixed_days_mask],
        np.array(daily_means)[mixed_days_mask],
        color="#e74c3c", marker='o', label='Hohe intra-tägliche Schwankung', s=80, zorder=5, edgecolors='white', linewidth=1.5
    )
    
    style_plot(ax, "Analyse intra-täglicher Stimmungsschwankungen (Mixed States)")
    ax.set_ylabel('Stimmungswert')
    ax.legend(loc='upper left')
    st.pyplot(fig)
    st.download_button("Download Plot", fig_to_bytes(fig), "mixed_states.png")
    
# (Der Rest der App-Logik in main() bleibt weitgehend gleich)
def main():
    st.title("Daylio Stimmungsanalyse")
    st.write("Laden Sie Ihren Daylio-Export (CSV) hoch, um modernisierte Visualisierungen und Frühwarnsignale für Ihre Stimmungsdynamik zu erhalten.")
    
    uploaded_file = st.file_uploader("Daylio CSV-Datei hochladen", type=["csv"])

    if uploaded_file:
        st.sidebar.header("Einstellungen für die Analyse")
        win = st.sidebar.slider("Roll. Fenster (Varianz/Autokorr) [Tage]", 7, 180, 90)
        entropy_win = st.sidebar.slider("Entropie-Fenster [Tage]", 7, 180, 60)
        sg_win = st.sidebar.slider("Savitzky-Golay Glättung [Tage]", 7, 61, 31, step=2)
        loess_frac = st.sidebar.slider("LOESS Glättung (Anteil Datenpunkte)", 0.05, 0.25, 0.08)

        df_raw = load_and_preprocess_data(uploaded_file)
        df_daily, df_intraday = calculate_metrics(df_raw, win, entropy_win, loess_frac, sg_win)

        # Plot-Sektionen
        st.subheader("Verteilung & Zeitverlauf")
        col1, col2 = st.columns([1, 2])
        with col1:
            plot_histogram(df_daily)
        with col2:
            plot_mood_timeseries(df_daily)

        st.subheader("Langfristige Trends & Frühwarnsignale")
        plot_smoothing(df_daily)
        plot_early_warning_signals(df_daily, win)
        plot_entropy(df_daily, entropy_win)
        
        st.subheader("Analyse der Tages-Dynamik")
        plot_mixed_states(df_intraday)
        n_intra = int(df_intraday['Mixed_IntraDay'].sum())
        st.metric(label="Tage mit hoher intra-täglicher Varianz", value=f"{n_intra}", delta=f"{n_intra/len(df_intraday):.1%} der Tage")

    else:
        st.info("Bitte laden Sie eine Daylio-Export-CSV-Datei hoch, um die Analyse zu starten.")

if __name__ == '__main__':
    main()