import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

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
- Rolling Varianz & Autokorrelation (Frühwarnsignale)
- Stimmungsglättung (Raw, Savitzky-Golay, LOESS)
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

    # Rolling/Glättungs-Einstellungen
    st.sidebar.header("Einstellungen für Rolling-Fenster & Glättung")
    win = st.sidebar.slider("Rolling-Fenster (Tage)", min_value=7, max_value=180, value=90, step=1)
    loess_frac = st.sidebar.slider("LOESS Glättung (Fraktion)", 0.05, 0.5, 0.15)

    # Frühwarnsignale berechnen
    df_tagesmittel['Varianz'] = df_tagesmittel['Stimmungswert'].rolling(window=win).var()
    df_tagesmittel['Autokorr'] = df_tagesmittel['Stimmungswert'].rolling(window=win).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False)

    # Plot 1: Frühwarnsignale
    st.subheader("Überlagerte Frühwarnsignale: Varianz vs. Autokorrelation")
    fig1, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(df_tagesmittel['Datum'], df_tagesmittel['Varianz'], color='gold', label=f'Rolling Varianz ({win} Tage)')
    ax1.plot(df_tagesmittel['Datum'], df_tagesmittel['Autokorr'], color='orange', label=f'Rolling Autokorrelation (Lag 1, {win} Tage)')
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
        "Beide Werte gelten als mögliche 'Frühwarnsignale' für einen Stimmungsumschwung."
    )

    # Stimmungsglättung: Raw, Savitzky-Golay, LOESS
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
        "Geglättete Linien helfen, langfristige Muster und Trendwechsel leichter zu erkennen, während tägliche Ausreißer weniger Gewicht bekommen."
    )

    # Automatische Stimmungsklassifikation (nur für Balkendiagramm)
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

    # Verteilung als Balkendiagramm
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