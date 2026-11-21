import streamlit as st
import requests
import pandas as pd
from google import genai
from google.genai import types

# Configuración optimizada para pantalla de smartphone
st.set_page_config(page_title="Fútbol Live +EV", page_icon="⚽", layout="centered")

# Credenciales seguras desde Streamlit Secrets
API_FOOTBALL_KEY = st.secrets["API_FOOTBALL_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_FOOTBALL_KEY}

# IDs de ligas objetivo en API-Football v3
TOP_LEAGUES = {
    39: "Premier League",
    140: "LaLiga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
    2: "Champions League",
    40: "Championship",
    88: "Eredivisie",
}


def get_live_fixtures(only_top_leagues: bool = True) -> list:
    """Obtiene los partidos en juego (consume 1 petición)."""
    resp = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={"live": "all"}, timeout=15)
    resp.raise_for_status()
    matches = resp.json().get("response", [])

    if only_top_leagues:
        matches = [m for m in matches if m["league"]["id"] in TOP_LEAGUES]
    return matches


def get_match_details(fixture_id: int) -> tuple[list, list]:
    """Obtiene estadísticas y eventos clave de un partido (consume 2 peticiones)."""
    stats_resp = requests.get(
        f"{BASE_URL}/fixtures/statistics", headers=HEADERS, params={"fixture": fixture_id}, timeout=15
    )
    events_resp = requests.get(
        f"{BASE_URL}/fixtures/events", headers=HEADERS, params={"fixture": fixture_id}, timeout=15
    )
    stats_resp.raise_for_status()
    events_resp.raise_for_status()
    return stats_resp.json().get("response", []), events_resp.json().get("response", [])


def format_stats_table(stats_raw: list) -> pd.DataFrame | None:
    """Convierte el JSON de estadísticas en una tabla limpia para móvil."""
    if len(stats_raw) < 2:
        return None

    home_name = stats_raw[0]["team"]["name"]
    away_name = stats_raw[1]["team"]["name"]

    home_dict = {item["type"]: item["value"] for item in stats_raw[0]["statistics"]}
    away_dict = {item["type"]: item["value"] for item in stats_raw[1]["statistics"]}

    key_metrics = [
        "Ball Possession",
        "Total Shots",
        "Shots on Goal",
        "Shots insidebox",
        "Corner Kicks",
        "Yellow Cards",
        "Red Cards",
        "expected_goals",
    ]

    rows = []
    for metric in key_metrics:
        if metric in home_dict or metric in away_dict:
            rows.append({
                "Métrica": metric,
                home_name: home_dict.get(metric, 0) or 0,
                away_name: away_dict.get(metric, 0) or 0,
            })

    return pd.DataFrame(rows)


def analyze_with_gemini(match_info: dict, stats_df: pd.DataFrame | None, events: list, market: str, odds: float, bankroll: int) -> str:
    """Envía el contexto real del partido a Gemini para el análisis cuantitativo."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Filtrar solo goles y tarjetas rojas para no saturar el prompt
    key_events = [
        f"Min {e['time']['elapsed']}' - {e['team']['name']}: {e['type']} ({e['detail']}) - {e['player']['name']}"
        for e in events
        if e["type"] in ("Goal", "Card") and ("Red" in str(e["detail"]) or e["type"] == "Goal")
    ]

    stats_text = stats_df.to_markdown(index=False) if stats_df is not None else "Estadísticas no disponibles aún."
    events_text = "\n".join(key_events) if key_events else "Sin goles ni tarjetas rojas registradas."

    prompt = f"""
Actúa como un analista cuantitativo de apuestas deportivas en vivo. Evalúa el siguiente partido con rigor matemático (sin inventar datos):

### Datos del Partido en Vivo
- **Liga:** {match_info['league']}
- **Partido:** {match_info['home']} {match_info['goals_home']} - {match_info['goals_away']} {match_info['away']}
- **Minuto / Estado:** {match_info['elapsed']}' ({match_info['status']})

### Estadísticas Oficiales en Vivo
{stats_text}

### Goles y Tarjetas Rojas
{events_text}

### Apuesta a Evaluar
- **Mercado:** {market}
- **Cuota ofrecida:** {odds}
- **Bankroll actual:** ${bankroll:,} CLP

### Entregable Requerido
1. **Lectura del partido (máx. 3 puntos):** Quién domina realmente según tiros al arco, volumen en área y xG, y cómo impacta el marcador/minuto actual.
2. **Modelo Probabilístico:** Estima la probabilidad real ($P$) para el tiempo restante y calcula la **Cuota Justa** ($1/P$).
3. **Valor Esperado ($EV$):** Calcula $\\text{{EV}} = (P \\times {odds}) - 1$ en porcentaje.
4. **Veredicto y Stake:**
   - Si $EV \\le 0\\%$, indica explícitamente **NO APOSTAR ($0 CLP)**.
   - Si $EV > 0\\%$, calcula el **Quarter-Kelly** (1/4 de Kelly) con tope estricto del **3% de la banca** (${int(bankroll * 0.03):,} CLP) y entrega el monto exacto en CLP.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return response.text


# ================= INTERFAZ DE USUARIO =================
st.title("⚽ Analizador Live +EV")

# Configuración de Bankroll y Filtro
col1, col2 = st.columns(2)
with col1:
    bankroll = st.number_input("Bankroll (CLP)", min_value=1000, value=100000, step=5000)
with col2:
    st.write("")  # Espaciado visual
    only_top = st.toggle("Solo Ligas Top", value=True)

# Botón manual para evitar gasto innecesario de cuota API
if st.button("🔄 Cargar partidos en vivo", use_container_width=True):
    with st.spinner("Consultando API-Football..."):
        try:
            st.session_state["live_matches"] = get_live_fixtures(only_top_leagues=only_top)
            if not st.session_state["live_matches"]:
                st.warning("No hay partidos en vivo en este momento con el filtro seleccionado.")
        except Exception as e:
            st.error(f"Error al consultar API-Football: {e}")

# Selector de partido si ya hay partidos cargados
if "live_matches" in st.session_state and st.session_state["live_matches"]:
    matches = st.session_state["live_matches"]

    options = {}
    for m in matches:
        label = (
            f"[{m['fixture']['status']['elapsed']}' - {m['fixture']['status']['short']}] "
            f"{m['teams']['home']['name']} {m['goals']['home']}-{m['goals']['away']} {m['teams']['away']['name']} "
            f"({m['league']['name']})"
        )
        options[label] = m

    selected_label = st.selectbox("Selecciona el partido:", list(options.keys()))
    selected_match = options[selected_label]

    st.divider()

    # Datos de la apuesta en tu casa
    c1, c2 = st.columns([2, 1])
    with c1:
        market_input = st.text_input("Mercado a evaluar", placeholder="Ej: Más de 2.5 goles, Gana Local, Cash Out...")
    with c2:
        odds_input = st.number_input("Tu Cuota", min_value=1.01, max_value=50.0, value=1.50, step=0.01)

    if st.button("🧠 Analizar con Gemini (+EV)", type="primary", use_container_width=True):
        if not market_input.strip():
            st.warning("Escribe el mercado que quieres evaluar antes de analizar.")
        else:
            fixture_id = selected_match["fixture"]["id"]
            match_info = {
                "league": selected_match["league"]["name"],
                "home": selected_match["teams"]["home"]["name"],
                "away": selected_match["teams"]["away"]["name"],
                "goals_home": selected_match["goals"]["home"],
                "goals_away": selected_match["goals"]["away"],
                "elapsed": selected_match["fixture"]["status"]["elapsed"],
                "status": selected_match["fixture"]["status"]["short"],
            }

            with st.spinner("Extrayendo estadísticas y calculando probabilidades..."):
                try:
                    stats_raw, events = get_match_details(fixture_id)
                    stats_df = format_stats_table(stats_raw)

                    if stats_df is not None:
                        st.subheader("📊 Estadísticas en Vivo")
                        st.dataframe(stats_df, hide_index=True, use_container_width=True)

                    analysis = analyze_with_gemini(
                        match_info, stats_df, events, market_input, odds_input, bankroll
                    )
                    st.subheader("📈 Diagnóstico Cuantitativo")
                    st.markdown(analysis)

                except Exception as e:
                    st.error(f"Ocurrió un error en el análisis: {e}")
