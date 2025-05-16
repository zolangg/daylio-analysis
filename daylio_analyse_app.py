import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

# --- ADAPTIVE SCHWELLENWERT-FUNKTIONEN ---
def varianz_warnschwelle(win):
    return max(0.2, 0.4 - 0.0015 * win)
def varianz_kritschwelle(win):
    return max(0.3, 0.5 - 0.001 * win)
def autokorr_warnschwelle(win):
    return 0.45 - 0.0005 * win
def autokorr_kritschwelle(win):
    return 0.65 - 0.0003 * win
def shannon_warnschwelle(win):
    return 1.3 + 0.001 * win
def shannon_kritschwelle(win):
    return 1.6 + 0.001 * win
def apen_warnschwelle(win):
    return 0.5 + 0.001 * win
def apen_kritschwelle(win):
    return 0.7 + 0.0015 * win

# --- ENTROPIE-HILFSFUNKTIONEN ---
def shannon_entropy(sequence):
    values, counts = np.unique(sequence, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return entropy

def approximate_entropy(U, m=2, r=0.2):
    U = np.array(U)
    N = len(U)
    if N <= m+1:
        return np.nan
    def _phi(m):
        X = np.array([U[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r, axis=0) / (N - m + 1.0)
        return np.sum(np.log(C + 1e-12)) / (N - m + 1.0)
    return abs(_phi(m) - _phi(m+1))

def fig1_to_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

st.set_page_config(layout="wide", page_title="Daylio Stimmungsanalyse")
st.title("Daylio Stimmungsanalyse & Frühwarnsignale")
st.write("""
Lade deinen Daylio-Export (CSV) hoch und erhalte die wichtigsten Visualisierungen:
- Rolling Varianz & Autokorrelation (Frühwarnsignale, adaptive Schwellen)
- Stimmungsglättung (Raw, Savitzky-Golay, LOESS)
- Entropie-Maße als Stabilitätsindikator (adaptive Schwellen)
- Verteilung der Stimmungs-Kategorien als Balkendiagramm
""")

uploaded_file = st.file_uploader("Daylio CSV-Datei hochladen", type=["csv"])

if uploaded_file:
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

    # Tagesmittel berechnen
    df_tagesmittel = df.groupby('full_date')['Stimmungswert'].mean().reset_index()
    df_tagesmittel = df_tagesmittel.sort_values('full_date')
    df_tagesmittel['Datum'] = df_tagesmittel['full_date']

    # Sidebar für alle Fenstergrößen
    st.sidebar.header("Einstellungen")
    win = st.sidebar.slider("Rolling-Fenster (Varianz/Autokorr)", min_value=7, max_value=180, value=90, step=1)
    loess_frac = st.sidebar.slider("LOESS Glättung (Fraktion)", 0.05, 0.5, 0.15)
    entropy_win = st.sidebar.slider("Entropie-Fenster (Tage)", min_value=7, max_value=180, value=60, step=1)

    # --- Frühwarnsignale ---
    df_tagesmittel['Varianz'] = df_tagesmittel['Stimmungswert'].rolling(window=win).var()
    df_tagesmittel['Autokorr'] = df_tagesmittel['Stimmungswert'].rolling(window=win).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False)

    st.subheader("Überlagerte Frühwarnsignale: Varianz vs. Autokorrelation")
    fig1, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(df_tagesmittel['Datum'], df_tagesmittel['Varianz'], color='gold', label=f'Rolling Varianz ({win} Tage)')
    ax1.plot(df_tagesmittel['Datum'], df_tagesmittel['Autokorr'], color='orange', label=f'Rolling Autokorrelation (Lag 1, {win} Tage)')
    # --- Adaptive Baselines ---
    ax1.axhline(varianz_warnschwelle(win), color='red', linestyle='--', alpha=0.6)
    ax1.text(df_tagesmittel['Datum'].iloc[5], varianz_warnschwelle(win)+0.02, f"Varianz Warnsignal ({varianz_warnschwelle(win):.2f})", color='red', fontsize=9, va='bottom')
    ax1.axhline(varianz_kritschwelle(win), color='crimson', linestyle=':', alpha=0.6)
    ax1.text(df_tagesmittel['Datum'].iloc[5], varianz_kritschwelle(win)+0.02, f"Varianz kritisch ({varianz_kritschwelle(win):.2f})", color='crimson', fontsize=9, va='bottom')
    ax1.axhline(autokorr_warnschwelle(win), color='blue', linestyle='--', alpha=0.6)
    ax1.text(df_tagesmittel['Datum'].iloc[5], autokorr_warnschwelle(win)+0.02, f"Autokorr Warnsignal ({autokorr_warnschwelle(win):.2f})", color='blue', fontsize=9, va='bottom')
    ax1.axhline(autokorr_kritschwelle(win), color='navy', linestyle=':', alpha=0.6)
    ax1.text(df_tagesmittel['Datum'].iloc[5], autokorr_kritschwelle(win)+0.02, f"Autokorr kritisch ({autokorr_kritschwelle(win):.2f})", color='navy', fontsize=9, va='bottom')
    ax1.set_title("Überlagerte Frühwarnsignale: Varianz vs. Autokorrelation")
    ax1.set_xlabel("Datum")
    ax1.set_ylabel("Wert")
    ax1.legend()
    st.pyplot(fig1)
    st.download_button("Download Plot 1 als PNG", data=fig1_to_bytes(fig1), file_name="fruehwarnsignale.png")
    st.caption(
        "**Interpretation:**\n"
        "• Die goldene Linie zeigt, wie stark deine Stimmung über ein gleitendes Fenster schwankt (Varianz). "
        "Steigt die Varianz, gibt es größere Stimmungsschwankungen.\n"
        "• Die orange Linie zeigt die Autokorrelation – sie misst, wie stark deine Stimmung an aufeinanderfolgenden Tagen ähnlich bleibt. "
        "Ein starker Anstieg der Autokorrelation kann auf eine beginnende Phase (z. B. manisch oder depressiv) hindeuten. "
        "Die Baselines passen sich automatisch an die Fenstergröße an und markieren empirisch fundierte Warn- und Kritisch-Schwellen."
    )

    # --- Stimmungsglättung ---
    st.subheader("Stimmungsglättung")
    raw = df_tagesmittel['Stimmungswert'].values
    x = np.arange(len(raw))
    try:
        sg = savgol_filter(raw, window_length=min(31, len(raw) // 2 * 2 + 1), polyorder=3)
    except Exception:
        sg = np.full_like(raw, np.nan)
    loess = lowess(raw, x, frac=loess_frac, return_sorted=False)

    fig2, ax2 = plt.subplots(figsize=(12,5))
    ax2.plot(df_tagesmittel['Datum'], raw, color='gold', alpha=0.3, label="Tagesmittel (roh)")
    ax2.plot(df_tagesmittel['Datum'], sg, color='orange', label="Savitzky-Golay")
    ax2.plot(df_tagesmittel['Datum'], loess, color='crimson', label="LOESS")
    # --- Baselines Stimmung (fix, da nicht fensterabhängig) ---
    ax2.axhline(2.5, color='royalblue', linestyle='--', alpha=0.6)
    ax2.text(df_tagesmittel['Datum'].iloc[5], 2.5+0.05, "Schwelle Depression (2.5)", color='royalblue', fontsize=9, va='bottom')
    ax2.axhline(3.5, color='darkgreen', linestyle='--', alpha=0.6)
    ax2.text(df_tagesmittel['Datum'].iloc[5], 3.5+0.05, "Schwelle Hypomanie (3.5)", color='darkgreen', fontsize=9, va='bottom')
    ax2.set_title("Stimmungsglättung")
    ax2.set_xlabel("Datum")
    ax2.set_ylabel("Stimmungswert")
    ax2.legend()
    st.pyplot(fig2)
    st.download_button("Download Plot 2 als PNG", data=fig1_to_bytes(fig2), file_name="stimmungsglaettung.png")
    st.caption(
        "**Interpretation:**\n"
        "Diese Grafik zeigt den geglätteten Verlauf deiner Stimmung über die Zeit. "
        "Die goldene Linie ist der rohe Tagesmittelwert, die orange und rote Linien zeigen geglättete Trends (Savitzky-Golay und LOESS). "
        "Die blauen/grünen Baselines markieren die klinisch relevanten Schwellen (depressiv <2.5, hypoman/manisch >3.5)."
    )

    # --- Entropie-Berechnung ---
    st.subheader("Shannon Entropie & Approximate Entropie (Stabilität der Stimmung)")
    shannon_entropies = []
    apen_entropies = []
    data = df_tagesmittel['Stimmungswert'].values
    for i in range(len(data)):
        if i < entropy_win - 1:
            shannon_entropies.append(np.nan)
            apen_entropies.append(np.nan)
        else:
            window = data[i - entropy_win + 1 : i + 1]
            shannon_entropies.append(shannon_entropy(window))
            std_win = np.std(window)
            apen_entropies.append(approximate_entropy(window, m=2, r=0.2 * std_win if std_win > 0 else 0.2))

    df_tagesmittel['Shannon Entropy'] = shannon_entropies
    df_tagesmittel['Approximate Entropy'] = apen_entropies

    fig4, ax4 = plt.subplots(figsize=(12,5))
    ax4.plot(df_tagesmittel['Datum'], df_tagesmittel['Shannon Entropy'], label='Shannon Entropie', color='blue')
    ax4.plot(df_tagesmittel['Datum'], df_tagesmittel['Approximate Entropy'], label='Approximate Entropy', color='red')
    # --- Adaptive Baselines Entropie ---
    ax4.axhline(shannon_warnschwelle(entropy_win), color='gray', linestyle='--', alpha=0.6)
    ax4.text(df_tagesmittel['Datum'].iloc[5], shannon_warnschwelle(entropy_win)+0.03, f"Shannon Warnsignal ({shannon_warnschwelle(entropy_win):.2f})", color='gray', fontsize=9, va='bottom')
    ax4.axhline(shannon_kritschwelle(entropy_win), color='black', linestyle=':', alpha=0.6)
    ax4.text(df_tagesmittel['Datum'].iloc[5], shannon_kritschwelle(entropy_win)+0.03, f"Shannon kritisch ({shannon_kritschwelle(entropy_win):.2f})", color='black', fontsize=9, va='bottom')
    ax4.axhline(apen_warnschwelle(entropy_win), color='purple', linestyle='--', alpha=0.6)
    ax4.text(df_tagesmittel['Datum'].iloc[5], apen_warnschwelle(entropy_win)+0.03, f"ApEn Warnsignal ({apen_warnschwelle(entropy_win):.2f})", color='purple', fontsize=9, va='bottom')
    ax4.axhline(apen_kritschwelle(entropy_win), color='magenta', linestyle=':', alpha=0.6)
    ax4.text(df_tagesmittel['Datum'].iloc[5], apen_kritschwelle(entropy_win)+0.03, f"ApEn kritisch ({apen_kritschwelle(entropy_win):.2f})", color='magenta', fontsize=9, va='bottom')
    ax4.set_title(f"Stabilität der Stimmung: Shannon & Approximate Entropy ({entropy_win}-Tage-Fenster)")
    ax4.set_xlabel("Datum")
    ax4.set_ylabel("Entropie")
    ax4.legend()
    st.pyplot(fig4)
    st.download_button("Download Plot 3 als PNG", data=fig1_to_bytes(fig4), file_name="entropie.png")
    st.caption(
        "**Interpretation:**\n"
        "• Die **Shannon Entropie** (blau) misst, wie unterschiedlich und unvorhersehbar deine Stimmung im Zeitfenster ist. "
        "Hohe Werte bedeuten viele unterschiedliche Stimmungen, niedrige Werte stehen für Gleichförmigkeit und Stabilität.\n"
        "• Die **Approximate Entropy** (rot) bewertet die Komplexität und Vorhersagbarkeit deines Stimmungsverlaufs. "
        "Niedrige Werte bedeuten wiederholbare, stabile Muster, hohe Werte zeigen chaotische, schwer vorhersagbare Verläufe.\n"
        "Die adaptiven Baselines markieren Wertebereiche, die laut Studienlage auffällig ('Warnsignal') oder kritisch sind."
    )

    # --- Automatische Stimmungsklassifikation für Balkendiagramm ---
    def mood_class(value):
        if pd.isna(value):
            return "Keine Daten"
        elif value <= 1.5:
            return "Super Low"
        elif value <= 2.5:
            return "Low"
        elif value <= 3.5:
            return "Euthym"
        elif value <= 4.5:
            return "High"
        else:
            return "Super High"

    df_tagesmittel["Stimmungs-Kategorie"] = df_tagesmittel["Stimmungswert"].apply(mood_class)

    st.subheader("Verteilung der Stimmungs-Kategorien (Tagesmittel, Gesamtzeitraum)")
    mood_cat_counts = df_tagesmittel["Stimmungs-Kategorie"].value_counts().reindex(
        ["Super Low", "Low", "Euthym", "High", "Super High"], fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.bar(mood_cat_counts.index, mood_cat_counts.values, color=['#8e44ad', '#3498db', '#f1c40f', '#e67e22', '#e74c3c'])
    ax3.set_ylabel("Tage")
    ax3.set_xlabel("Kategorie")
    ax3.set_title("Verteilung der Stimmungs-Kategorien (nach Tagesmittel)")
    st.pyplot(fig3)
    st.caption(
        "**Interpretation:**\n"
        "Hier siehst du, wie viele Tage du in welcher Stimmungs-Kategorie verbracht hast (z. B. 'Low', 'Euthym', 'High'). "
        "So kannst du auf einen Blick erkennen, welche Stimmungslagen in deinem Verlauf dominieren und wie häufig extreme Phasen auftreten."
    )

else:
    st.info("Bitte lade zuerst eine Daylio-Export-CSV hoch.")
