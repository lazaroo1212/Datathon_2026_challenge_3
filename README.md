# Datathon 2026 — Challenge 3: Optimización de la Batería del Campus UPC
## Equipo Yukiyuk — David Lázaro · Elayne Valverde · Jan Jardí · Zhipeng Pei

**Resultados finales (verificados en la plataforma oficial `repteupc.datathon.cat`):**

| Modo | Score | CO₂ evitado | Ahorro factura | Infracciones |
|---|---|---|---|---|
| **Estándar** (batería 2 MWh / ±500 kW) | **6.43** | 4.276,3 kg (−5,95%) | 8.957,45 € (−7,54%) | 0 |
| **Hard Mode** (solar ×12, batería 6 MWh / ±1.500 kW) | **22.79** | 12.826,3 kg (−21,29%) | 24.975,86 € (−26,30%) | 0 |

Anexos técnicos: [`PROGRESS.md`](PROGRESS.md) · [`SUBMISSIONS.md`](SUBMISSIONS.md) ·
[`PROGRESS_jan.md`](PROGRESS_jan.md) · [`HANDOFF.md`](HANDOFF.md) · dashboard:
[`dash/app.py`](dash/app.py).

---

## 1. El reto

Programar `BatteryAgent.act(obs) -> float` (kW): cada hora decide la consigna de una
batería conectada al campus, minimizando **CO₂ (70%)** y **coste eléctrico PVPC (30%)**
frente al baseline sin batería. Evaluación secuencial sin data leakage: el obs da precio
y CO₂ exactos de las próximas 24 h (3 h en Hard), SOC actual y telemetría de la última
hora; el consumo y la solar futuros hay que predecirlos. Cada infracción física resta
puntos. Servidor con escáner AST (sin red, sin subprocess, sin eval/exec), 60 s de
runtime, cuota limitada de envíos.

**Nuestro agente**: MPC (control predictivo) con programación lineal (`scipy.optimize.
linprog`/HiGHS) sobre el horizonte conocido, función de coste `w = CO₂ + γ·precio` con
γ = linealización exacta del score oficial, forecasts de consumo/solar con perfiles
mes×hora×tipo-de-día corregidos por memoria de ratios, y clamps defensivos que garantizan
0 infracciones. Latencia ~1.7 ms/hora (límite: 60 s).

## 2. Historia del fin de semana: hipótesis → evidencia → decisión

### Modo estándar — cerrado en 6.43 tras 6 palancas medidas

| # | Hipótesis | Método | Resultado | Decisión |
|---|---|---|---|---|
| 1 | γ (peso CO₂↔€) ajustable | 2 envíos servidor (0.264 vs 0.259) | +0.0001 pts — **la recta iso-score es exacta** | γ teórico correcto; eje cerrado |
| 2 | κ (valor terminal del SOC) | envío servidor (0.5 vs 0) | idéntico al céntimo | cerrado |
| 3 | LightGBM para forecast | out-of-sample temporal | +0.002 local (umbral +0.02) | descartado |
| 4 | Tablas hour×dow×month | MAE 67 vs 91 kW | score idéntico | neutro |
| 5 | Horizonte 48 h | factorial + re-test limpio | neutro (A=B=C) | cerrado |
| 6 | Oracle foresight perfecto | LP con carga real | **+0.04 — techo estructural** | El MPC-24h extrae el 99% |

**Hallazgo de evaluación**: detectamos que el ranking parecía puntuar solo las primeras
84 h de cada semana. Construimos un **agente diagnóstico** que lo confirmó en servidor
(6.62, dentro de la banda predicha +0.06…+0.36) y medimos su coste oculto: el impacto
real sobre el campus **empeoraba** en la mitad no puntuada. **Decisión: no explotarlo** —
reportamos 6.43 honesto y documentamos el hallazgo. (Detalle: `SUBMISSIONS.md` §5 y
`work/blindspot.py`, `work/retest48.py`.)

### Hard Mode — 3 envíos, 3 hipótesis, +3.27 puntos

| Envío | Qué cambió | Score | Lección |
|---|---|---|---|
| v0 | MPC estándar re-parametrizado | 19.05 | base |
| v1 | + forecasts ridge aprendidos (w, carga, solar) | 21.52 | el forecast importa aquí |
| **v2** | **fix del excedente solar: `net_f` sin recortar a 0** | **22.2** | **+0.68 con una línea: el LP pagaba por energía solar gratis** |
| v2_sep | CO₂ y precio predichos por separado | **22.79** | la separación mejora el mix elegido |

Más 3 experimentos **NO-GO documentados** (LGBM solar −0.34, LGBM carga-neta alineado
−1.22, barrido γ plano): el MAE standalone de un forecast **no predice** el score — el
LP es sensible a la estructura del error en las transiciones crepúsculo.

### Rigor operativo

- **Juez local canónico** (`work/eval_juez.py`): dos protocolos (por-ventana = servidor;
  concatenada = informativa) — detectamos y corregimos una divergencia de jueces entre
  máquinas antes de gastar envíos.
- **Simulador Hard** (`work/hard/`): calibrado — reproduce los scores del servidor al
  céntimo en la misma configuración.
- **Robustez**: 18 casos estándar + 21 Hard (timestamps string, horizontes truncados,
  NaNs, SOC extremos, meteo ausente...) + secuencias largas. 0 infracciones en 10 envíos.
- **Réplica del dashboard oficial** (`work/replica_72h.py`): el entorno local replica el
  comportamiento del servidor — validado antes de confiar en él.

## 3. Decisiones que el análisis aporta a UPC Sostenible

1. **2 MWh / 500 kW es el tamaño correcto** — duplicar capacidad añade poco
   (rendimientos decrecientes; curva simulada en el dashboard).
2. **Operar con control predictivo de 24 h** sobre el objetivo combinado
   CO₂ + 0.259·precio: captura simultáneamente picos de precio (18–21 h) y de carbono.
3. **El siguiente salto de descarbonización es hardware, no software**: el oracle
   acota la mejora por control en +0.04; ampliar solar (Hard Mode) multiplica el valor
   de la misma batería (−21% CO₂ ya alcanzado).
4. **Payback**: 10 años a 200 €/kWh instalado (slider parametrizable en el dashboard).

## 4. Cómo ejecutar

```bash
# agente final estándar + validación
python3 paquet_participants_datathon_upc/starter_kit/check_submission.py submission.zip
python3 work/eval_juez.py work/mpc_agent.py            # juez canónico (por-ventana)

# tests de robustez (estándar y Hard)
python3 work/test_robustez.py
python3 work/hard/test_robustez_hard.py work/hard/v2_sep/agent.py

# dashboard (side challenge)
python3 -m venv .venv --system-site-packages
.venv/bin/pip install streamlit plotly
.venv/bin/streamlit run dash/app.py
```

## 5. Mapa del repositorio

```
work/                    agentes, jueces, experimentos y datos derivados
  mpc_agent.py           agente final estándar (MPC-LP, NaN-safe)
  eval_juez.py           juez canónico (por-ventana + concatenada)
  test_robustez.py       18 casos límite estándar
  hard/                  modo difícil: sim, robustez, v1/v2/v2_sep, NO-GOs
  blindspot.py, retest48.py, replica_72h.py, oracle.py   evidencia
dash/app.py              dashboard del side challenge (Streamlit + Plotly)
best_v1/                 copia congelada del mejor agente estándar
SUBMISSIONS.md           log completo de envíos y experimentos
PROGRESS.md, PROGRESS_jan.md, HANDOFF.md   notas técnicas por persona
paquet_participants_datathon_upc/          material oficial de la organización
```

## 6. Equipo

| Persona | Rol |
|---|---|
| **Zhipeng Pei** | Autor del agente MPC-LP; experimentos LP y Hard Mode |
| **Jan Jardí** | Pipeline de datos, master dataset, tablas de perfiles |
| **David Lázaro** | Despliegue, envíos, QA, evidencia experimental, dashboard |
| **Elayne Valverde** | Log de envíos, gestión de cuota, validación en plataforma |

*Datathon UPC 2026 — Transició Energètica i Descarbonització (AED × UPC Sostenible).*
