"""Battery Ops Cockpit — Dashboard del side challenge (David).

4 pestañas = 4 decisiones de UPC Sostenible:
  1. Inversión   -> ¿comprar la batería? KPIs oficiales + what-if tamaño/potencia
  2. Operación   -> ¿cómo opera? SOC/consigna vs demanda, heatmap ahorro hora×mes
  3. Trade-off   -> ¿CO₂ vs €? recta iso-score + curva gamma simulada
  4. Evidencia   -> ¿qué sabemos? tabla de experimentos cerrados + techo oracle

Todo simulado EN VIVO con el agente MPC real (work/mpc_agent.py) sobre hourly.csv
(ventanas pseudo-test días 18–24 de cada mes), con st.cache_data.

Ejecutar:  .venv/bin/streamlit run dash/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "work"))
sys.path.insert(0, str(ROOT / "paquet_participants_datathon_upc" / "starter_kit"))

from environment import CampusBatteryEnv  # noqa: E402
from mpc_agent import BatteryAgent  # noqa: E402
import mpc_agent  # noqa: E402

# ── constantes oficiales del servidor ──────────────────────────────────────
CO2_BASE = 71816.0        # kg CO2, 12 semanas test
COST_BASE = 118746.34     # EUR, 12 semanas test
GAMMA_DEF = 0.264

st.set_page_config(page_title="Battery Ops Cockpit — Yukiyuk", layout="wide")
st.title("⚡ Battery Ops Cockpit — UPC Sostenible")
st.caption("Dashboard de decisión del Challenge 3 · equipo Yukiyuk · datos reales del campus "
           "(ventanas pseudo-test días 18–24 de cada mes, agente MPC-LP real)")


# ── carga de datos y simulación (caché) ────────────────────────────────────
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "work" / "hourly.csv", parse_dates=["Fecha"]).set_index("Fecha")
    return df.dropna(subset=["precio_eur_per_kwh", "co2_kg_per_kwh"])


@st.cache_data
def semanas_disponibles(df: pd.DataFrame) -> list[str]:
    out = []
    for (y, m), g in df.groupby([df.index.year, df.index.month]):
        w = g[(g.index.day >= 18) & (g.index.day <= 24)]
        if len(w) >= 100:
            out.append(f"{y}-{m:02d}")
    return out


def simular_semana(df: pd.DataFrame, etiqueta: str, cap: float = 2000.0,
                   pmax: float = 500.0, gamma: float | None = None) -> dict:
    """Corre el agente MPC real sobre la semana (días 18-24) y devuelve métricas+series."""
    y, m = int(etiqueta[:4]), int(etiqueta[5:7])
    win = df[(df.index.year == y) & (df.index.month == m)
             & (df.index.day >= 18) & (df.index.day <= 24)].copy()
    mpc_agent.CAP = cap
    mpc_agent.PMAX = pmax
    agent = BatteryAgent()
    if gamma is not None:
        agent.gamma = gamma
    spec = {"capacity_kwh": cap, "power_kw": pmax, "efficiency": 0.9,
            "soc_init": 0.5, "soc_min": 0.0, "soc_max": 1.0, "dt_hours": 1.0}
    env = CampusBatteryEnv(win.reset_index(), spec=spec)
    obs = env.reset()
    done = False
    while not done:
        p = agent.act(obs)
        obs, done = env.step(p)
    res = env.evaluate_results()
    load = win["load_kw"].to_numpy()
    pv = win["pv_kw"].to_numpy()
    base_grid = np.maximum(load - pv, 0.0)
    grid = np.asarray(env.grid)
    price = win["precio_eur_per_kwh"].to_numpy()
    co2f = win["co2_kg_per_kwh"].to_numpy()
    ahorro_eur_h = (base_grid - grid) * price          # >=0 cuando desplazamos importación
    return {
        "res": res,
        "fechas": win.index,
        "load": load, "pv": pv, "grid": grid, "base_grid": base_grid,
        "soc": np.asarray(env.soc_series),
        "consigna": np.asarray(env.actions),
        "price": price, "co2f": co2f,
        "ahorro_eur_h": ahorro_eur_h,
        "ahorro_co2_h": (base_grid - grid) * co2f,
    }


df = cargar_datos()
semanas = semanas_disponibles(df)

tab1, tab2, tab3, tab4 = st.tabs(
    ["💰 1 · ¿Comprar la batería?", "🔧 2 · ¿Cómo opera?",
     "⚖️ 3 · ¿CO₂ o euros?", "🔬 4 · ¿Qué sabemos?"])

# ══════════════════════════ TAB 1 · INVERSIÓN ══════════════════════════════
with tab1:
    st.subheader("Caso de inversión: batería de 2 MWh operada por MPC")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score oficial", "6.43 pts", "0 infracciones")
    c2.metric("CO₂ evitado (12 sem)", "4.276 kg", "−5,95% vs baseline")
    c3.metric("Factura ahorrada", "8.957 €", "−7,54% vs baseline")
    c4.metric("Runtime servidor", "3,4 s", "de 60 s permitidos")

    st.divider()
    st.markdown("**What-if: ¿y si la batería fuera más pequeña o más potente?** "
                "(se re-simula la semana seleccionada con el mismo agente)")
    cc1, cc2, cc3 = st.columns([1, 1, 2])
    semana_sel = cc1.selectbox("Semana (días 18–24)", semanas, index=semanas.index("2025-09") if "2025-09" in semanas else 0)
    cap_sel = cc2.select_slider("Capacidad (kWh)", options=[500, 1000, 2000, 4000], value=2000)
    pmax_sel = cc3.select_slider("Potencia (kW)", options=[250, 500, 1000], value=500)

    with st.spinner("Simulando con el agente MPC real..."):
        s_base = simular_semana(df, semana_sel)                     # 2 MWh / 500 kW
        s_alt = simular_semana(df, semana_sel, cap=cap_sel, pmax=pmax_sel)

    def oficial(res: dict) -> tuple[float, float, float]:
        dco2, dcost = res["co2_estalviat_pct"], res["cost_estalviat_pct"]
        return 0.7 * dco2 + 0.3 * dcost, dco2, dcost

    score_b, co2_b, cost_b = oficial(s_base["res"])
    score_a, co2_a, cost_a = oficial(s_alt["res"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Score semana (2 MWh/500 kW)", f"{score_b:.2f}", f"CO₂ −{co2_b:.2f}% · € −{cost_b:.2f}%")
    m2.metric(f"Score semana ({cap_sel/1000:g} MWh/{pmax_sel} kW)", f"{score_a:.2f}",
              f"{'▲' if score_a >= score_b else '▼'} {abs(score_a-score_b):.2f} vs 2 MWh",
              delta_color="off" if score_a >= score_b else "inverse")
    m3.metric("CO₂ evitada semana", f"{s_alt['res']['co2_estalviat_kg']:.0f} kg")
    m4.metric("€ ahorrados semana", f"{s_alt['res']['cost_estalviat_eur']:.0f} €")

    # curva de rendimientos decrecientes (capacidad) sobre la semana seleccionada
    with st.spinner("Curva capacidad→ahorro (5 simulaciones)..."):
        caps = [500, 1000, 2000, 4000]
        pts = []
        for c in caps:
            r = simular_semana(df, semana_sel, cap=c, pmax=500)["res"]
            pts.append({"capacidad_kwh": c, "ahorro_eur": r["cost_estalviat_eur"],
                        "co2_kg": r["co2_estalviat_kg"]})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p["capacidad_kwh"] for p in pts],
                             y=[p["ahorro_eur"] for p in pts],
                             name="€ ahorrados/semana", mode="lines+markers"))
    fig.add_vline(x=2000, line_dash="dot", annotation_text="batería propuesta (2 MWh)")
    fig.update_layout(title="Rendimientos decrecientes: más batería ≠ más ahorro proporcional "
                            "(la demanda no lo aprovecha)",
                      xaxis_title="Capacidad (kWh)", yaxis_title="€ ahorrados en la semana")
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Decisión 1**: 2 MWh/500 kW captura la mayor parte del ahorro alcanzable; "
               "duplicar capacidad añade poco — el límite lo pone la estructura de demanda "
               "(~29 MWh/día vs 2 MWh de batería).")

# ══════════════════════════ TAB 2 · OPERACIÓN ══════════════════════════════
with tab2:
    st.subheader(f"Operación de la batería — semana {semana_sel}")
    f1, f2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_base["fechas"], y=s_base["load"] - s_base["pv"],
                             name="Demanda neta (sin batería)", line=dict(color="#6ba3d6")))
    fig.add_trace(go.Scatter(x=s_base["fechas"], y=s_base["grid"],
                             name="Importación con batería", line=dict(color="#1f4e79")))
    fig.add_trace(go.Scatter(x=s_base["fechas"], y=s_base["consigna"],
                             name="Consigna batería (kW)", line=dict(color="#19a974"), opacity=0.7))
    fig.update_layout(title="Balanço de potencia: la batería recorta los picos de importación",
                      yaxis_title="kW", height=420)
    f1.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=s_base["fechas"], y=100 * s_base["soc"] / 2000,
                              name="SOC %", line=dict(color="#0e7c5a")))
    fig2.add_hline(y=100, line_dash="dot", annotation_text="2.000 kWh")
    fig2.add_hline(y=0, line_dash="dot", annotation_text="0 kWh")
    fig2.update_layout(title="Ciclo diario: carga a mediodía (barato/limpio), descarga en picos 18–21h",
                       yaxis_title="SOC (%)", height=420)
    f2.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("**¿Cuándo se gana el dinero? Mapa hora × mes del ahorro en factura**")
    with st.spinner("Simulando 10-11 semanas para el heatmap..."):
        filas = []
        for et in semanas:
            s = simular_semana(df, et)
            hh = pd.DataFrame({"hora": s["fechas"].hour, "mes": s["fechas"].month,
                               "eur": s["ahorro_eur_h"], "co2": s["ahorro_co2_h"]})
            filas.append(hh.groupby(["mes", "hora"])[["eur", "co2"]].mean().reset_index())
    hm = pd.concat(filas).groupby(["mes", "hora"]).mean().reset_index()
    pivot = hm.pivot(index="hora", columns="mes", values="eur")
    fig3 = px.imshow(pivot, aspect="auto", color_continuous_scale="Greens",
                     labels=dict(x="mes", y="hora del día", color="€/h ahorrados"),
                     title="Ahorro medio por hora y mes — el valor vive en los picos 18–21h")
    st.plotly_chart(fig3, use_container_width=True)
    st.info("**Decisión 2**: operar la batería como MPC rodante de 24 h maximiza el recorte "
            "de picos vespertinos; el verano y los findes aportan menos (demanda plana) — "
            "no hace falta sobredimensionar para esas estaciones.")

# ══════════════════════════ TAB 3 · TRADE-OFF ══════════════════════════════
with tab3:
    st.subheader("El trade-off CO₂ ↔ coste: una recta, no un área")
    st.markdown("Score = 0.7·ΔCO₂ + 0.3·ΔCost. Mover el peso γ (€→kg-CO₂) **desliza el "
                "reparto a lo largo de la recta iso-score pero no aumenta el total** — "
                "lo verificamos en el servidor con dos envíos reales (6.43 en ambos).")

    lb = pd.DataFrame({  # snapshot real del leaderboard 14/11
        "equipo": ["Programi qui programi (1º)", "Prueba_1", "La Datoneta", "Manguets",
                   "Los Rojos", "Viquipèdia racers", "JAMM", "Psycode", "La MasIA", "Yukiyuk"],
        "dco2": [5.99, 5.96, 5.97, 5.95, 5.96, 5.97, 5.97, 5.97, 5.95, 5.95],
        "dcost": [7.48, 7.57, 7.50, 7.55, 7.52, 7.50, 7.50, 7.50, 7.54, 7.54],
    })
    lb["score"] = 0.7 * lb["dco2"] + 0.3 * lb["dcost"]
    fig = px.scatter(lb, x="dcost", y="dco2", text="equipo", color="score",
                     color_continuous_scale="Viridis",
                     labels={"dcost": "ΔCost %", "dco2": "ΔCO₂ %"})
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(title="Leaderboard real 14/11: todos los 6.43 sobre la misma recta iso-score",
                      height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Simulación local: barrido de γ sobre la semana seleccionada** "
                "(cada punto = semana completa re-simulada)")
    gammas = [0.05, 0.10, 0.18, 0.264, 0.35, 0.50]
    with st.spinner("Barrido de γ (6 simulaciones)..."):
        pts = []
        for g in gammas:
            r = simular_semana(df, semana_sel, gamma=g)["res"]
            pts.append({"gamma": g, "dco2": r["co2_estalviat_pct"],
                        "dcost": r["cost_estalviat_pct"],
                        "score": 0.7 * r["co2_estalviat_pct"] + 0.3 * r["cost_estalviat_pct"]})
    gp = pd.DataFrame(pts)
    figg = px.line(gp, x="dcost", y="dco2", markers=True, text="gamma",
                   labels={"dcost": "ΔCost %", "dco2": "ΔCO₂ %"},
                   title=f"Semana {semana_sel}: γ desliza el punto, el score se mantiene")
    figg.update_traces(textposition="top center")
    figg.add_annotation(x=gp["dcost"].iloc[-1], y=gp["dco2"].iloc[-1],
                        text="γ alto = prioriza €", showarrow=True, arrowhead=1)
    figg.add_annotation(x=gp["dcost"].iloc[0], y=gp["dco2"].iloc[0],
                        text="γ bajo = prioriza CO₂", showarrow=True, arrowhead=1)
    st.plotly_chart(figg, use_container_width=True)
    st.success("**Decisión 3**: el trade-off no es una decisión de diseño — γ = "
               "0.3·CO₂_base / (0.7·Cost_base) = 0.259 es la linealización exacta del score. "
               "La única forma de salir de la recta es una batería/estrategia mayor.")

# ══════════════════════════ TAB 4 · EVIDENCIA ══════════════════════════════
with tab4:
    st.subheader("Todo lo probado, con evidencia")
    ev = pd.DataFrame([
        ["γ (peso CO₂↔€)", "Servidor: envíos 1 vs 3", "+0.0001 pts — recta iso-score exacta", "Cerrado"],
        ["κ (valor terminal SOC)", "Servidor: envío 4 vs 1", "Idéntico al céntimo", "Cerrado"],
        ["β / β_mem (corrección forecast)", "Sweep local", "Plano 0–0.5", "Cerrado"],
        ["LightGBM t+1..t+24", "Pseudo-test local", "+0.002 (umbral +0.02) — LP insensible", "Descartado"],
        ["Tablas hour×dow×month", "Pseudo-test local", "MAE 67 vs 91 kW pero score idéntico", "Neutro"],
        ["Horizonte 48h + perfil w", "Re-test limpio días 11–17", "A=B=C=5.534 — neutro", "NO-GO"],
        ["5 mecanismos de reserva 48h", "sweep48/reserve/gate", "Todos ≤0 — cicla 2.000 kWh/día", "NO-GO"],
        ["Oracle foresight perfecto", "LP con carga real", "+0.04 — techo estructural", "Bound"],
        ["Robustez", "18 casos límite + 72h", "NaN-safe, 0 infracciones", "OK"],
    ], columns=["Palanca", "Cómo se probó", "Resultado", "Estado"])
    st.dataframe(ev, use_container_width=True, height=380)

    c1, c2 = st.columns(2)
    c1.info("**Techo del control**: con la batería cíclica (2.000 kWh/día) y demanda ≫ batería, "
            "el MPC-24h ya extrae el ~99% del valor alcanzable (oracle +0.04). "
            "Más inteligencia de control ≠ más puntos.")
    c2.warning("**Implicación de compra**: la palanca real de descarbonización no es más "
               "software, es más batería o más solar — el dashboard lo cuantifica en la "
               "pestaña 1 (curva de rendimientos decrecientes).")

    st.divider()
    st.markdown("**Infraestructura validada**: réplica del dashboard oficial del servidor "
                "(mismo patrón de ciclos), 0 infracciones en 4 envíos reales, agente a prueba "
                "de datos sucios (NaN, horizontes truncados, timestamps string).")

# ── nota metodológica al pie ───────────────────────────────────────────────
st.caption("Metodología: simulaciones locales con environment.py del starter kit (física "
           "validada contra el servidor) sobre hourly.csv; KPIs de cabecera = resultados "
           "oficiales de la plataforma (4 envíos reales, 14/11/2026). γ = conversión "
           "€→kg-CO₂ implícita en el score oficial.")
