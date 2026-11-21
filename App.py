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


@st.cache_data(ttl=3600)
def get_available_gemini_models() -> list[str]:
    """Consulta los modelos Gemini habilitados en tu cuenta (cacheado 1 hora)."""
    fallback = ["gemini-3.8-flash"]
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=15000),
        )
        models = []
        excluded = ("embedding", "aqa", "imagen", "veo", "tts", "audio", "robotics", "vision")
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            lower_name = name.lower()
            if "gemini" not in lower_name or any(word in lower_name for word in excluded):
                continue
            actions = getattr(m, "supported_actions", None) or []
            if actions and "generateContent" not in actions:
                continue
            models.append(name)

        if not models:
            return fallback

        models = sorted(set(models), reverse=True)
        if "gemini-3.8-flash" in models:
            models.remove("gemini-3.8-flash")
            models.insert(0, "gemini-3.8-flash")
        return models
    except Exception:
        return fallback


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


def stream_gemini(prompt: str, model_name: str):
    """Genera la respuesta de Gemini en streaming con timeout de 60 segundos."""
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=60000),
    )
    response_stream = client.models.generate_content_stream(
        model=model_name,
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

# Selector automático de modelo Gemini
available_models = get_available_gemini_models()
with st.expander("⚙️ Modelo de IA (cambiar si hay saturación)", expanded=False):
    selected_model = st.selectbox(
        "Modelo API de Gemini:",
        options=available_models,
        index=0,
    )
    st.caption("Si un modelo arroja 503 o demora demasiado, cambia a otra versión Flash/Lite aquí o usa el botón de copiado para tu app de Gemini Pro.")

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

        b1, b2 = st.columns(2)
        with b1:
            run_ev_api = st.button("🧠 Analizar aquí (API)", type="primary", use_container_width=True)
        with b2:
            gen_ev_copy = st.button("📋 Copiar a Gemini Pro", use_container_width=True)

        if run_ev_api:
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
                    st.subheader(f"📈 Diagnóstico ({selected_model})")
                    st.write_stream(stream_gemini(prompt, selected_model))
                except Exception as e:
                    st.error(f"La API no respondió a tiempo o está saturada: {e}")
                    st.info("Prueba cambiar el modelo arriba en ⚙️ Modelo de IA, o toca '📋 Copiar a Gemini Pro' para analizarlo al instante en tu app.")

        if gen_ev_copy:
            if not market_input.strip():
                st.warning("Escribe el mercado que quieres evaluar antes de generar el texto.")
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
                    st.success("Toca el ícono de copiar (arriba a la derecha de la caja) y pégalo en tu app de Gemini Pro:")
                    st.code(prompt, language="markdown")
                except Exception as e:
                    st.error(f"Error al cargar estadísticas: {e}")

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

        cb1, cb2 = st.columns(2)
        with cb1:
            run_co_api = st.button("💰 Evaluar aquí (API)", type="primary", use_container_width=True)
        with cb2:
            gen_co_copy = st.button("📋 Copiar a Gemini Pro", key="co_copy_btn", use_container_width=True)

        if run_co_api:
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
                    st.subheader(f"💰 Diagnóstico Cash Out ({selected_model})")
                    st.write_stream(stream_gemini(prompt, selected_model))
                except Exception as e:
                    st.error(f"La API no respondió a tiempo o está saturada: {e}")
                    st.info("Prueba cambiar el modelo en ⚙️ Modelo de IA, o toca '📋 Copiar a Gemini Pro'.")

        if gen_co_copy:
            if not active_market.strip():
                st.warning("Indica cuál es tu apuesta activa antes de generar el texto.")
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
                    st.success("Toca el ícono de copiar (arriba a la derecha de la caja) y pégalo en tu app de Gemini Pro:")
                    st.code(prompt, language="markdown")
                except Exception as e:
                    st.error(f"Error al cargar estadísticas: {e}")
