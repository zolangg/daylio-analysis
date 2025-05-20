import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
import networkx as nx
import seaborn as sns

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
- Häufigkeitsverteilung in 0.5er-Schritten und farbcodierter Mood Zeitverlauf
- Stimmungsglättung (Raw, Savitzky-Golay, LOESS)
- Rolling Varianz & Autokorrelation (Frühwarnsignale, adaptive Schwellen)
- Entropie-Maße als Stabilitätsindikator (adaptive Schwellen)
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
    loess_frac = st.sidebar.slider("LOESS Glättung (Fraktion)", 0.05, 0.20, 0.08)
    entropy_win = st.sidebar.slider("Entropie-Fenster (Tage)", min_value=7, max_value=180, value=60, step=1)

    # --- Frühwarnsignale ---
    df_tagesmittel['Varianz'] = df_tagesmittel['Stimmungswert'].rolling(window=win).var()
    df_tagesmittel['Autokorr'] = df_tagesmittel['Stimmungswert'].rolling(window=win).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False)

    # --- Entropie-Berechnung ---
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

    # --- WARN-POPUP (Kritische Schwellen überschritten?) ---
    warnungen = []

    akt_var = df_tagesmittel['Varianz'].iloc[-1]
    akt_aut = df_tagesmittel['Autokorr'].iloc[-1]
    akt_shannon = df_tagesmittel['Shannon Entropy'].iloc[-1]
    akt_apen = df_tagesmittel['Approximate Entropy'].iloc[-1]

    if akt_var > varianz_kritschwelle(win):
        warnungen.append(f"**Varianz kritisch:** {akt_var:.2f} > {varianz_kritschwelle(win):.2f} (Instabilität!)")
    if akt_aut > autokorr_kritschwelle(win):
        warnungen.append(f"**Autokorrelation kritisch:** {akt_aut:.2f} > {autokorr_kritschwelle(win):.2f} (Persistente Stimmungslage/Frühwarnsignal)")
    if akt_shannon > shannon_kritschwelle(entropy_win):
        warnungen.append(f"**Shannon Entropie kritisch:** {akt_shannon:.2f} > {shannon_kritschwelle(entropy_win):.2f} (Stimmung sehr chaotisch)")
    if akt_apen > apen_kritschwelle(entropy_win):
        warnungen.append(f"**Approximate Entropy kritisch:** {akt_apen:.2f} > {apen_kritschwelle(entropy_win):.2f} (Unregelmäßiger Verlauf)")

    if warnungen:
        st.error("🚨 **KRITISCHE WARNUNG:**\n\n" + "\n\n".join(warnungen))

    # --- PLOTS ---
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
        **Interpretation:** Dieses Histogramm zeigt die Verteilung deiner täglichen Stimmungsmittelwerte in feinen 0.5er-Schritten.  
        Du erkennst auf einen Blick, welche Stimmungslagen im Zeitverlauf besonders häufig waren und ob es Schwerpunkte im depressiven, euthymen oder (hypo)manischen Bereich gibt.
        """)

    # --- Mood Zeitverlauf (farbcodiert nach 0.5 Stufen) ---
    st.subheader("Mood Zeitverlauf")
    mood_vals = df_tagesmittel['Stimmungswert'].values
    dates = df_tagesmittel['Datum'].values


    def color_for_mood(val):
        if val <= 1.5:
            return '#8e44ad'  # Super Low
        elif val <= 2.0:
            return '#5e72b0'
        elif val <= 2.5:
            return '#3498db'  # Low
        elif val <= 3.0:
            return '#f1c40f'  # Euthym
        elif val <= 3.5:
            return '#e67e22'
        elif val <= 4.0:
            return '#e74c3c'  # High
        elif val <= 4.5:
            return '#c0392b'
        else:
            return '#900c3f'  # Super High


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
        **Interpretation:** In diesem Diagramm siehst du deinen gesamten Mood-Verlauf, wobei jeder Punkt nach seiner Stimmungskategorie (je 0.5 Punkte) eingefärbt ist.  
        So erkennst du auf einen Blick stabile Phasen, schnelle Stimmungswechsel, und wie lange bestimmte Zustände anhalten.
        Die horizontale Linien markieren die Übergänge zwischen den wichtigsten Stimmungsklassen.
        """)

    # --- Stimmungsglättung ---
    st.subheader("Stimmungsglättung")

    raw = df_tagesmittel['Stimmungswert'].values
    x = np.arange(len(raw))

    # Savitzky-Golay
    try:
        sg = savgol_filter(raw, window_length=min(31, len(raw) // 2 * 2 + 1), polyorder=3)
    except Exception:
        sg = np.full_like(raw, np.nan)

    # LOESS – Standard auf stärker geglättet (z. B. 0.08)
    loess_curve = lowess(raw, x, frac=loess_frac, return_sorted=False)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df_tagesmittel['Datum'], raw, color='gold', alpha=0.2, label="Tagesmittel (roh)")
    ax2.plot(df_tagesmittel['Datum'], sg, color='orange', linewidth=2, label="Savitzky-Golay")
    ax2.plot(df_tagesmittel['Datum'], loess_curve, color='crimson', linewidth=2, label="LOESS")

    # 0.5er-Stufen-Linien
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
    ax2.legend()
    st.pyplot(fig2)
    st.download_button(
        "Download Stimmungsglättung als PNG",
        data=fig1_to_bytes(fig2),
        file_name="stimmungsglaettung.png"
    )
    st.caption("""
        **Interpretation:**
        Die orange Linie ist die **Savitzky-Golay-Glättung**, die rote Linie die **LOESS-Glättung**.
        Die feinen grauen Linien markieren die 0.5er-Schritte zwischen den Stimmungszonen.
        """)

    st.subheader("Überlagerte Frühwarnsignale: Varianz vs. Autokorrelation")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df_tagesmittel['Datum'], df_tagesmittel['Varianz'], color='gold', label=f'Rolling Varianz ({win} Tage)')
    ax1.plot(df_tagesmittel['Datum'], df_tagesmittel['Autokorr'], color='orange',
             label=f'Rolling Autokorrelation (Lag 1, {win} Tage)')
    # --- Adaptive Baselines ---
    ax1.axhline(varianz_warnschwelle(win), color='grey', linestyle='--', alpha=0.4, linewidth=1)
    ax1.text(df_tagesmittel['Datum'].iloc[5], varianz_warnschwelle(win) + 0.02,
             f"Varianz Warnsignal ({varianz_warnschwelle(win):.2f})", color='gray', fontsize=9, va='bottom')
    ax1.axhline(varianz_kritschwelle(win), color='grey', linestyle=':', alpha=0.4, linewidth=1)
    ax1.text(df_tagesmittel['Datum'].iloc[5], varianz_kritschwelle(win) + 0.02,
             f"Varianz kritisch ({varianz_kritschwelle(win):.2f})", color='gray', fontsize=9, va='bottom')
    ax1.axhline(autokorr_warnschwelle(win), color='grey', linestyle='--', alpha=0.4, linewidth=1)
    ax1.text(df_tagesmittel['Datum'].iloc[5], autokorr_warnschwelle(win) + 0.02,
             f"Autokorr Warnsignal ({autokorr_warnschwelle(win):.2f})", color='gray', fontsize=9, va='bottom')
    ax1.axhline(autokorr_kritschwelle(win), color='grey', linestyle=':', alpha=0.4, linewidth=1)
    ax1.text(df_tagesmittel['Datum'].iloc[5], autokorr_kritschwelle(win) + 0.02,
             f"Autokorr kritisch ({autokorr_kritschwelle(win):.2f})", color='gray', fontsize=9, va='bottom')
    ax1.set_title("Überlagerte Frühwarnsignale: Varianz vs. Autokorrelation")
    ax1.set_xlabel("Datum")
    ax1.set_ylabel("Wert")
    ax1.legend()
    st.pyplot(fig1)
    st.download_button("Download Plot 1 als PNG", data=fig1_to_bytes(fig1), file_name="fruehwarnsignale.png")
    st.caption("""
        **Interpretation:**
        Die goldene Linie zeigt, wie stark deine Stimmung über ein gleitendes Fenster schwankt (**Varianz**).
        Steigt die **Varianz**, gibt es größere Stimmungsschwankungen.
        Die orange Linie zeigt die **Autokorrelation** – sie misst, wie stark deine Stimmung an aufeinanderfolgenden Tagen ähnlich bleibt.
        Ein starker Anstieg der **Autokorrelation** kann auf eine beginnende Phase (z. B. manisch oder depressiv) hindeuten.
        Die Baselines passen sich automatisch an die Fenstergröße an und markieren empirisch fundierte Warn- und Kritisch-Schwellen.
        """)

    st.subheader("Shannon Entropie & Approximate Entropie (Stabilität der Stimmung)")
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(df_tagesmittel['Datum'], df_tagesmittel['Shannon Entropy'], label='Shannon Entropie', color='blue')
    ax4.plot(df_tagesmittel['Datum'], df_tagesmittel['Approximate Entropy'], label='Approximate Entropy', color='red')
    # --- Adaptive Baselines Entropie ---
    ax4.axhline(shannon_warnschwelle(entropy_win), color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax4.text(df_tagesmittel['Datum'].iloc[5], shannon_warnschwelle(entropy_win) + 0.03,
             f"Shannon Warnsignal ({shannon_warnschwelle(entropy_win):.2f})", color='gray', fontsize=9, va='bottom')
    ax4.axhline(shannon_kritschwelle(entropy_win), color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax4.text(df_tagesmittel['Datum'].iloc[5], shannon_kritschwelle(entropy_win) + 0.03,
             f"Shannon kritisch ({shannon_kritschwelle(entropy_win):.2f})", color='gray', fontsize=9, va='bottom')
    ax4.axhline(apen_warnschwelle(entropy_win), color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax4.text(df_tagesmittel['Datum'].iloc[5], apen_warnschwelle(entropy_win) + 0.03,
             f"ApEn Warnsignal ({apen_warnschwelle(entropy_win):.2f})", color='gray', fontsize=9, va='bottom')
    ax4.axhline(apen_kritschwelle(entropy_win), color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax4.text(df_tagesmittel['Datum'].iloc[5], apen_kritschwelle(entropy_win) + 0.03,
             f"ApEn kritisch ({apen_kritschwelle(entropy_win):.2f})", color='gray', fontsize=9, va='bottom')
    ax4.set_title(f"Stabilität der Stimmung: Shannon & Approximate Entropy ({entropy_win}-Tage-Fenster)")
    ax4.set_xlabel("Datum")
    ax4.set_ylabel("Entropie")
    ax4.legend()
    st.pyplot(fig4)
    st.download_button("Download Plot 3 als PNG", data=fig1_to_bytes(fig4), file_name="entropie.png")
    st.caption("""
        **Interpretation:**
        Die **Shannon Entropie** (blau) misst, wie unterschiedlich und unvorhersehbar deine Stimmung im Zeitfenster ist.
        Hohe Werte bedeuten viele unterschiedliche Stimmungen, niedrige Werte stehen für Gleichförmigkeit und Stabilität.
        Die **Approximate Entropy** (rot) bewertet die Komplexität und Vorhersagbarkeit deines Stimmungsverlaufs.
        Niedrige Werte bedeuten wiederholbare, stabile Muster, hohe Werte zeigen chaotische, schwer vorhersagbare Verläufe.
        Die adaptiven Baselines markieren Wertebereiche, die laut Studienlage auffällig ('Warnsignal') oder kritisch sind.
        """)
    
    st.subheader("Markov Mood-Modell: Übergänge zwischen Stimmungszuständen")

    # --- Mood-Zustände und Label ---
    mood_labels = ['Super Low', 'Low', 'Euthym', 'High', 'Super High']

    # Zunächst die Stimmungswerte für Markov auf Integer-Stufen abbilden
    df_tagesmittel = df_tagesmittel.dropna(subset=['Stimmungswert'])
    mood_vals = df_tagesmittel['Stimmungswert'].round().astype(int).clip(1, 5)
    mood_idx = mood_vals - 1  # 0=Super Low, ..., 4=Super High

    # Übergangspaare berechnen
    transitions = list(zip(mood_idx[:-1], mood_idx[1:]))
    n_states = len(mood_labels)
    trans_matrix = np.zeros((n_states, n_states), dtype=int)
    for i, j in transitions:
        if 0 <= i < n_states and 0 <= j < n_states:
            trans_matrix[i, j] += 1

    # Zu Wahrscheinlichkeiten normieren (Zeile = Ausgangszustand)
    with np.errstate(invalid='ignore'):
        trans_prob = trans_matrix / trans_matrix.sum(axis=1, keepdims=True)

    # --- Plots nebeneinander (mit st.columns) ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Markov-Übergangsmatrix (Heatmap)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(trans_prob, annot=True, cmap="YlOrRd", fmt=".2f",
                xticklabels=mood_labels, yticklabels=mood_labels, cbar_kws={'label': 'Wahrscheinlichkeit'}, ax=ax)
        ax.set_xlabel("→ Nächster Zustand")
        ax.set_ylabel("Aktueller Zustand")
        ax.set_title("Markov-Übergangsmatrix")
        st.pyplot(fig)
        st.download_button(
            "Download Matrix als PNG",
            data=fig1_to_bytes(fig),
            file_name="markov_matrix.png"
        )

    with col2:
        st.markdown("**Markov State Diagram (Netzwerkplot)**")
        G = nx.DiGraph()
        for i, mood in enumerate(mood_labels):
            G.add_node(mood)
        for i in range(n_states):
            for j in range(n_states):
                prob = trans_prob[i, j]
                if prob > 0.05:
                    G.add_edge(mood_labels[i], mood_labels[j], weight=prob)
        pos = nx.circular_layout(G)
        edges = G.edges()
        weights = [6 * G[u][v]['weight'] for u, v in edges]
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        nx.draw_networkx_nodes(G, pos, node_size=1800, node_color="lightblue", ax=ax2)
        nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax2)
        nx.draw_networkx_edges(G, pos, width=weights, arrowsize=22, arrowstyle='-|>', ax=ax2)
        nx.draw_networkx_edge_labels(G, pos,
            edge_labels={(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}, font_size=10, ax=ax2)
        ax2.set_title("Markov State Diagram")
        ax2.axis('off')
        st.pyplot(fig2)
        st.download_button(
            "Download Netzwerk als PNG",
            data=fig1_to_bytes(fig2),
            file_name="markov_network.png"
        )
        st.caption("""
        Die linke Heatmap zeigt die Wahrscheinlichkeit, von jedem Zustand in einen anderen zu wechseln.  
        Der rechte Netzwerkplot zeigt die wichtigsten Übergänge als Pfeile. Dicke Pfeile und große Zahlen markieren häufige Übergänge.
        """)
else:
    st.info("Bitte lade zuerst eine Daylio-Export-CSV hoch.")
