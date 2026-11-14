# Datathon_2026_challenge_3

Repositorio del equipo para el Repte UPC Sostenible — Datathon 2026 (Challenge 3).
El detalle técnico del progreso está en [`PROGRESS.md`](PROGRESS.md).

**Members**: David Lázaro, Elayne Valverde, Jan Jardí, Zhipeng Pei

## Estado actual

- El agente MPC (`BatteryAgent`) fue **creado por Zhipeng Pei en su ordenador**:
  [`work/mpc_agent.py`](work/mpc_agent.py) + [`work/params.json`](work/params.json).
  Es la implementación de referencia del equipo (score local ~6.12, 0 infracciones, ~1.3 ms/paso).
- Pendiente: empaquetar `submission.zip`, validar con `check_submission.py` y hacer el primer envío a la plataforma.

## 👤 Reparto individual

### Pei — Autor del agente MPC + Forecasting

- **Autor del `BatteryAgent` MPC-LP** (`work/mpc_agent.py`): creado y desarrollado en su ordenador.
  Cualquier cambio al planificador (LP, horizonte, clamps, γ) pasa por él.
- Dos LightGBM/XGBoost: consumo_kw t+1..t+24 y solar_kw t+1..t+24.
- ⚠️ Validación temporal: entrenar con meses anteriores, testear con los últimos 7 días de cada mes (imita el test oculto). Nada de train_test_split aleatorio (leakage temporal).
- Guardar con joblib.dump. Target: MAE horario razonable; no perfeccionar de más, el optimizador tolera errores.

### David — Optimizador + Agent

- Cada hora: recoger `preus_properes_24h`, `co2_properes_24h` (¡son perfecto!), añadir predicciones de consumo/solar de P2, y resolver el plan óptimo de 24h:
- `scipy.optimize.linprog` (LP lineal por tramos) o greedy por ranking de horas "cargar en barato/limpio, descargar en caro/sucio".
- Ejecutar solo la primera hora del plan → horizonte rodante (MPC).
- Clipping defensivo obligatorio antes de devolver P_t: comprobar 0 ≤ SOC + P·η_c ≤ 2000 y SOC − |P|/η_d ≥ 0 con margen de seguridad (p. ej. usar límites efectivos 20–1980 kWh). Recordad η_c = η_d = √0.9 ≈ 0.9487.

### Jan — Datos y features

- Cargar los 6 CSV, resample 5min→1h (media), P_consum = max(0, −potencia_exportada), P_solar = suma de los 3 inversores.
- Filtrar anomalías (>5000 kW), imputar huecos (interpolación ≤15 min, media estacional el resto).
- Dataset maestro con features: hora, día semana, mes, festivos, lags 24/48/168h, rolling means, irradiancia, temperatura.

### Elayne — Validación y estrategia de submits

- `evaluate_local.py`: correr el agent sobre el `environment.py` local con datos reales de semanas test y calcular el score oficial exacto (0.7·ΔCO₂ + 0.3·ΔCost − infracciones).
- Curarse de errores: cada submit fallido o peor que el anterior = intento tirado.
- Registro de envíos + responsable del reloj (30 intentos en 3 turnos).

## ⏱️ Timeline con hitos

| Momento | Hito |
|---|---|
| H0–H2 | Todos leen starter_kit; Elayne corre tutorial.ipynb y sube el baseline del starter (asegura score en el servidor) |
| H2–H6 | Pipeline datos listo (Jan), modelos v1 (Pei), heurística por reglas en local (David), evaluador listo (Elayne) |
| H6 | Submit 2: heurística por reglas (validada en local) |
| Turno noche | MPC completo (Pei) + modelos afinados; 2–3 submits escalonados |
| Turno domingo | Robustez (NaN, huecos, timestamps raros), última pulida, reservar 2–3 intentos para el final |
