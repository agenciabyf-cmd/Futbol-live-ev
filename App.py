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
    resp = requests.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={"live": "all"},
        timeout=15,
    )
    resp.raise_for_status()
    matches = resp.json().get("response", [])

    if only_top_leagues:
        matches = [m for m in matches if m["league"]["id"] in TOP_LEAGUES]
    return matches


def get_match_details(fixture_id: int) -> tuple[list, list]:
    """Obtiene estadísticas y eventos clave de un partido (consume 2 peticiones)."""
    stats_resp = requests.get(
        f"{BASE_URL}/fixtures/statistics",
        headers=HEADERS,
        params={"fixture": fixture_id},
        timeout=15,
    )
    events_resp = requests.get(
        f"{BASE_URL}/fixtures/events",
        headers=HEADERS,
        params={"fixture": fixture_id},
        timeout=15,
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


def build_match_context(match_info: dict, stats_df: pd.DataFrame | None, events: list) -> str:
    """Formatea el estado del partido, estadísticas y eventos clave para el prompt."""
    key_events = [
        f"Min {e['time']['elapsed']}' - {e['team']['name']}: {e['type']} ({e['detail']}) - {e['player']['name']}"
        for e in events
        if e["type"] in ("Goal", "Card") and ("Red" in str(e["detail"]) or e["type"] == "Goal")
    ]

    stats_text = stats_df.to_markdown(index=False) if stats_df is not None else "Estadísticas no disponibles aún."
    events_text = "\n".join(key_events) if key_events else "Sin goles ni tarjetas rojas registradas."

    return f"""### Datos del Partido en Vivo
- **Liga:** {match_info['league']}
- **Partido:** {match_info['home']} {match_info['goals_home']} - {match_info['goals_away']} {match_info['away']}
- **Minuto / Estado:** {match_info['elapsed']}' ({match_info['status']})

### Estadísticas Oficiales en Vivo
{stats_text}

### Goles y Tarjetas Rojas
{events_text}"""


def stream_gemini(prompt: str):
    """Genera la respuesta de Gemini en streaming con timeout estricto de 25 segundos."""
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=25000),
    )
    response_stream = client.models.generate_content_stream(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    for chunk in response_stream:
        if chunk.text:
            yield chunk.text


def build_new_bet_prompt(
    match_info: dict,
    stats_df: pd.DataFrame | None,
    events: list,
    market: str,
    odds: float,
    bankroll: int,
) -> str:
    """Construye el prompt para evaluar +EV y Quarter-Kelly."""
    context = build_match_context(match_info, stats_df, events)
    max_stake = int(bankroll * 0.03)

    return f"""Actúa como un analista cuantitativo de apuestas deportivas en vivo. Evalúa el siguiente partido con rigor matemático (sin inventar datos):

{context}

### Nueva Apuesta a Evaluar
- **Mercado:** {market}
- **Cuota ofrecida:** {odds}
- **Bankroll actual:** ${bankroll:,} CLP

### Entregable Requerido
1. **Lectura del partido (máx. 3 puntos):** Quién domina realmente según tiros al arco, volumen en área y xG, y cómo impacta el marcador/minuto actual.
2. **Modelo Probabilístico:** Estima la probabilidad real ($P$) para el tiempo restante y calcula la **Cuota Justa** ($1/P$).
3. **Valor Esperado ($EV$):** Calcula $\\text{{EV}} = (P \\times {odds}) - 1$ en porcentaje.
4. **Veredicto y Stake:**
   - Si $EV \\le 0\\%$, indica explícitamente **NO APOSTAR ($0 CLP)**.
   - Si $EV > 0\\%$, calcula el **Quarter-Kelly** (1/4 de Kelly) con tope estricto del **3% de la banca** (${max_stake:,} CLP) y entrega el monto exacto en CLP.
"""


def build_cashout_prompt(
    match_info: dict,
    stats_df: pd.DataFrame | None,
    events: list,
    active_market: str,
    stake: int,
    original_odds: float,
    cashout_offer: int,
) -> str:
    """Construye el prompt para evaluar Cash Out vs. mantener posición."""
    context = build_match_context(match_info, stats_df, events)
    potential_payout = int(round(stake * original_odds))

    return f"""Actúa como un analista cuantitativo de gestión de riesgo en apuestas deportivas en vivo. Evalúa si conviene tomar el Cash Out o mantener la posición activa:

{context}

### Posición Activa del Usuario
- **Apuesta activa:** {active_market}
- **Stake invertido:** ${stake:,} CLP
- **Cuota original:** {original_odds}
- **Pago potencial si gana:** ${potential_payout:,} CLP
- **Oferta de Cash Out actual:** ${cashout_offer:,} CLP

### Entregable Requerido
1. **Lectura de riesgo en cancha (máx. 3 puntos):** Evalúa si el trámite actual (xG, tiros al arco, presión, tarjetas, minuto) favorece o amenaza la apuesta activa.
2. **Probabilidad real de cobro ($P$):** Estima la probabilidad real de que la apuesta termine ganando en el tiempo restante.
3. **Comparativa Financiera Exacta:**
   - **Valor Esperado de Mantener (Hold EV):** $P \\times {potential_payout:,}$ CLP.
   - **Oferta de Cash Out:** ${cashout_offer:,} CLP.
   - **Diferencia (Comisión implícita de la casa):** $\\text{{Hold EV}} - \\text{{Cash Out}}$ en CLP y porcentaje.
4. **Veredicto Claro:** Indica explícitamente **MANTENER POSICIÓN** o **TOMAR CASH OUT**, justificando la decisión desde la esperanza matemática y el riesgo de varianza en los minutos finales.
"""


def ensure_stats_loaded(fixture_id: int) -> dict:
    """Carga las estadísticas solo si no están ya guardadas para este partido."""
    cached = st.session_state.get("cached_stats")
    if cached is not None and cached["fixture_id"] == fixture_id:
        return cached

    stats_raw, events = get_match_details(fixture_id)
    stats_df = format_stats_table(stats_raw)
    st.session_state["cached_stats"] = {
        "fixture_id": fixture_id,
        "stats_df": stats_df,
        "events": events,
    }
    return st.session_state["cached_stats"]


# ================= INTERFAZ DE USUARIO =================
st.title("⚽ Analizador Live +EV")

col1, col2 = st.columns(2)
with col1:
    bankroll = st.number_input("Bankroll (CLP)", min_value=1000, value=100000, step=5000)
with col2:
    st.write("")
    only_top = st.toggle("Solo Ligas Top", value=True)

if st.button("🔄 Cargar partidos en vivo", use_container_width=True):
    with st.spinner("Consultando API-Football..."):
        try:
            st.session_state["live_matches"] = get_live_fixtures(only_top_leagues=only_top)
            st.session_state.pop("cached_stats", None)
            if not st.session_state["live_matches"]:
                st.warning("No hay partidos en vivo en este momento con el filtro seleccionado.")
        except Exception as e:
            st.error(f"Error al consultar API-Football: {e}")

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

    cached = st.session_state.get("cached_stats")
    has_current_stats = cached is not None and cached["fixture_id"] == fixture_id

    btn_label = "📊 Actualizar estadísticas del partido" if has_current_stats else "📊 Cargar estadísticas del partido"
    if st.button(btn_label, use_container_width=True):
        with st.spinner("Descargando estadísticas en vivo..."):
            try:
                stats_raw, events = get_match_details(fixture_id)
                stats_df = format_stats_table(stats_raw)
                st.session_state["cached_stats"] = {
                    "fixture_id": fixture_id,
                    "stats_df": stats_df,
                    "events": events,
                }
                has_current_stats = True
            except Exception as e:
                st.error(f"Error al obtener estadísticas: {e}")

    if has_current_stats:
        stats_df = st.session_state["cached_stats"]["stats_df"]
        if stats_df is not None:
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
        else:
            st.info("API-Football aún no publica estadísticas detalladas para este partido.")

    st.divider()

    tab_ev, tab_cashout = st.tabs(["📈 Nueva Apuesta (+EV)", "💰 Evaluar Cash Out"])

    with tab_ev:
        c1, c2 = st.columns([2, 1])
        with c1:
            market_input = st.text_input(
                "Mercado a evaluar",
                placeholder="Ej: Más de 2.5 goles, Gana Local...",
                key="ev_market",
            )
        with c2:
            odds_input = st.number_input(
                "Tu Cuota",
                min_value=1.01,
                max_value=50.0,
                value=1.50,
                step=0.01,
                key="ev_odds",
            )

        if st.button("🧠 Analizar Nueva Apuesta (+EV)", type="primary", use_container_width=True):
            if not market_input.strip():
                st.warning("Escribe el mercado que quieres evaluar antes de analizar.")
            else:
                try:
                    cached_now = ensure_stats_loaded(fixture_id)
                    prompt = build_new_bet_prompt(
                        match_info,
                        cached_now["stats_df"],
                        cached_now["events"],
                        market_input,
                        odds_input,
                        bankroll,
                    )
                    st.subheader("📈 Diagnóstico Cuantitativo")
                    st.write_stream(stream_gemini(prompt))
                except Exception as e:
                    st.error(f"Gemini no respondió a tiempo o está saturado: {e}")
                    st.info("Tus estadísticas ya están guardadas en memoria. Espera 10 segundos y vuelve a tocar el botón sin gastar peticiones de API-Football.")

    with tab_cashout:
        active_market = st.text_input(
            "Tu apuesta activa",
            placeholder="Ej: Manchester City gana, Más de 1.5 goles...",
            key="co_market",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            stake_input = st.number_input(
                "Stake apostado (CLP)",
                min_value=100,
                value=3000,
                step=500,
                key="co_stake",
            )
            orig_odds = st.number_input(
                "Cuota original",
                min_value=1.01,
                max_value=100.0,
                value=1.50,
                step=0.01,
                key="co_odds",
            )
        with col_b:
            potential_win = int(round(stake_input * orig_odds))
            st.metric("Pago potencial si gana", f"${potential_win:,} CLP")
            cashout_offer = st.number_input(
                "Oferta Cash Out (CLP)",
                min_value=0,
                value=int(stake_input),
                step=100,
                key="co_offer",
            )

        if st.button("💰 Evaluar Cash Out con Gemini", type="primary", use_container_width=True):
            if not active_market.strip():
                st.warning("Indica cuál es tu apuesta activa antes de evaluar el Cash Out.")
            else:
                try:
                    cached_now = ensure_stats_loaded(fixture_id)
                    prompt = build_cashout_prompt(
                        match_info,
                        cached_now["stats_df"],
                        cached_now["events"],
                        active_market,
                        stake_input,
                        orig_odds,
                        cashout_offer,
                    )
                    st.subheader("💰 Diagnóstico de Cash Out")
                    st.write_stream(stream_gemini(prompt))
                except Exception as e:
                    st.error(f"Gemini no respondió a tiempo o está saturado: {e}")
                    st.info("Tus estadísticas ya están guardadas en memoria. Espera 10 segundos y vuelve a tocar el botón sin gastar peticiones de API-Football.")
