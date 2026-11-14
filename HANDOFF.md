# HANDOFF — Contexto completo del equipo Yukiyuk (Datathon UPC 2026, Challenge 3)

> Documento de traspaso generado el sábado 14/11/2026 ~18:00. Léelo antes de tocar nada.

## 1. El reto (30 segundos)

- Programar `BatteryAgent.act(obs)` (Python, 1 clase) que cada hora decide la consigna
  de una **batería de 2.000 kWh / 500 kW / η_ida-vuelta 90% (η_c=η_d=√0.9≈0.9487)** del
  campus UPC, para minimizar **CO₂ (70%)** y **coste PVPC (30%)** vs. baseline sin batería.
- Score oficial: `0.7·ΔCO₂% + 0.3·ΔCost% − 1.0·infracciones` (el env local `environment.py`
  penaliza solo 0.002/infracción — **5× más blando que el oficial**; nosotros: 0 siempre).
- Test oculto: últimos 7 días de cada mes (2.016 h). Cada hora el obs da: precio y CO₂
  **exactos de las próximas 24 h**, SOC, y consumo/solar de la hora pasada. El futuro de
  consumo y solar NO se conoce (hay que forecastar o aproximar).
- Plataforma: `repteupc.datathon.cat` — Validación Rápida (gratis) y Trametre al
  Leaderboard (consume cuota). **30 envíos total** (10×3 turnos). Límites: 60 s runtime,
  25 MB zip. Equipo: **Yukiyuk**, PIN definido el día 1.

## 2. Estado actual (sábado 14/11 ~18:00)

| Concepto | Valor |
|---|---|
| **Mejor score servidor** | **6.43** (ΔCO₂ 5.95%, ΔCost 7.54%, 4.276,3 kg, 8.957,45 €, 0 infr.) |
| Envíos usados | **4/30** (1 mejor, 1 duplicado accidental, 2 experimentos) |
| Cuota restante | 6 hoy (hasta 21:00) + 10 noche + 10 mañana |
| Leaderboard | 21º en el pelotón 6.43; **líder 6.44 (=6.437 reales; nos faltan +0.009)** |
| Proxy local | 6.12 (ventanas días 18–24 de cada mes); offset servidor ≈ +0.31 |
| Baselines oficiales del test | 71.816 kg CO₂ / 118.746,34 € → **γ exacto = 0.259** |
| Agente canónico | `work/mpc_agent.py` + `work/params.json` (γ=0.264, κ=0, β=0.4, β_mem=0.6) |
| Copia congelada | `best_v1/` (score 6.43) — NO TOCAR |
| Paquete listo | `submission/` (agent.py + params.json) → `submission.zip` |

## 3. Agente: arquitectura MPC-LP (lo que puntúa)

Cada hora: construye `w[k] = co2[k] + γ·price[k]` para las 24 h conocidas (¡exactas!),
forecasta net-load con tablas `month×hour×weekday/weekend` (de `params.json`, generadas
por `build_tables.py`) + correcciones de memoria (β_mem) y última hora (β), y resuelve un
LP (`scipy.optimize.linprog`/HiGHS): variables carga/descarga/importación/SOC por hora,
objetivo `min Σ w·g`, con bisagra `g ≥ net_f + p_c − p_d` (los excedentes se pierden,
no se venden). Devuelve SOLO la primera hora (MPC rodante). Clamps defensivos con margen
0.02 kW antes de devolver (0 infracciones garantizadas). Runtime: ~1.7 ms/hora → ~3.4 s
de los 60 s permitidos. Helper `_clean()` sanitiza NaN de `noves_dades_5min` (fix 14/11).

## 4. TODO lo probado y CERRADO (no reintentar sin idea nueva)

| Palanca | Resultado | Evidencia |
|---|---|---|
| γ 0.264→0.259 (peso CO₂↔precio) | +0.0001 reales — **recta iso-score**, γ teórico exacto | envíos 1 vs 3 (kg/€ exactos) |
| κ terminal 0.5→0 | **idéntico al céntimo** (4.276,3 kg) | envío 4 vs 1 |
| β/β_mem sweeps | planos 0–0.5 | `work/sweep.py` |
| LightGBM forecast net-load t+1..24 | +0.002 local (umbral +0.02) — LP insensible a ±25 kW | `train_lgbm.py`, `eval_lgbm.py` |
| Tablas ricas 12×24×7 (hour×dow×month) | MAE 67 vs 91 kW pero **score idéntico** | `build_tables_rich.py` |
| Horizonte 48h + prof_w | −0.59 con artefacto de cola; **neutro (A=B=C=5.534) en re-test limpio** (días 11–17 con continuación real) | `retest48.py` |
| 48h: 5 mecanismos (extensión w, persistencia, reserva SOC fija/gated, valor terminal lejano) | **todos ≤0** — cicla 2.000 kWh/día; guardar para mañana cuesta igual que hoy | `sweep48.py`, `sweep_reserve.py`, `sweep_gate.py` |
| Blindspot (picos >24h) | cota 2.37 pts/año PERO **no capturable**: son anomalías que el perfil mensual no representa | `blindspot.py` + NO-GO Pei |
| Oracle foresight perfecto | solo +0.04 local (6.12→6.16) → **techo estructural del MPC-24h** | `oracle.py` |

**Síntesis**: el score lo limita la estructura del problema (batería 2 MWh vs carga
~29 MWh/día), no forecast, horizonte ni hiperparámetros. 6 las palancas medidas dan 0.

## 5. Validaciones de infraestructura (confía en ellas)

- `replica_72h.py`: réplica local del dashboard oficial — patrón idéntico (cargas +500
  12–15h, descargas −500 18–21h, ciclos SOC 0→100→0, 0 infr., score ventana 5.87) →
  **`eval_windows.py`/`environment.py` local es oráculo fiable del servidor**.
- `test_robustez.py`: 18 casos límite (timestamp string, H<24, H=1, NaNs, SOC extremos,
  arrays desiguales, consumo negativo...) + secuencia 72 pasos — **todo OK**.
- Dataset maestro de Jan: `master_dataset.csv` (8.761 h × 64 cols, festivos, lags,
  rollings, flags `in_hidden_week` / `*_imputed_frac`) — **0 mismatches vs hourly.csv**
  en filas reales. Trampa: franja 17 ago–1 sep 2026 imputada sin flag de semana oculta.

## 6. Reglas y trampas aprendidas (memorizar)

1. **Empaquetado**: `submission.zip` = `agent.py` + `params.json` EN LA RAÍZ. Cargar
   ficheros solo con `Path(__file__).parent`. Sin `os/sys/subprocess/eval/exec`
   (escáner AST del servidor; `check_submission.py` local lo replica).
2. **Workflow de envío**: cambios → `test_robustez.py` → `eval_windows.py` (umbral
   promoción ≥+0.02 local) → `check_submission.py` → Validació Ràpida (gratis) →
   Trametre. Un cambio por envío. Log en `SUBMISSIONS.md` (plantilla dentro).
3. **El timestamp llega como STRING** en el dry-run oficial — el agente ya lo maneja.
4. `environment.py` local trunca `preus_properes_24h` al final de los datos → cualquier
   experimento de horizonte debe evaluarse con continuación real (ver `retest48.py`).
5. Git: pull antes de editar docs compartidos; commits pequeños con el score en el
   mensaje; nunca sobrescribir `best_v1/`.
6. Ranking: dentro del mismo score, **gana el registro más temprano** → mejorar requiere
   +0.009 reales mínimo (tier 6.44 empieza en 6.437).

## 7. Equipo y reparto real

- **Pei**: autor del agente MPC (lo escribió en su portátil). Experimentos LP. NO-GO 48h.
- **Jan**: pipeline de datos + master dataset + tablas. `PROGRESS_jan.md`.
- **David**: despliegue, empaquetado, envíos, QA (blindspot, retest48, replica, robustez).
- **Elayne**: log de envíos (`SUBMISSIONS.md`), reloj de cuota, validación.
- Estado en `PROGRESS.md` (compartido) — mantenerlo al día.

## 8. Qué queda por hacer (valor esperado)

1. **Nada de enviar más sin idea genuinamente nueva** — todo lo medido da 0. El 6.43 ya
   está blindado (último envío = mejor score).
2. Ideas teóricamente abiertas (techo +0.04): control estocástico/robusto sobre la
   incertidumbre de carga, reoptimización de la reserva de fin de semana, MILP con modos.
   **Coste alto, ganancia máxima +0.04 — solo si sobra tiempo.**
3. Documentación final para el jurado (si puntúa): `PROGRESS.md`, `SUBMISSIONS.md`,
   notebook con la historia de experimentos (hipótesis→evidencia→decisión).
4. Mañana (Torn 3): si el leaderboard se mueve, mirar ΔCO₂/ΔCost de los rivales para
   inferir su estructura. Solo enviar si hay hipótesis ≥+0.02 local.

## 9. Mapa de ficheros clave

```
work/
  mpc_agent.py          ← AGENTE CANÓNICO (BatteryAgent MPC-LP, NaN-safe)
  params.json           ← γ=0.264, κ=0, β=0.4, β_mem=0.6, tablas 12×24×2
  hourly.csv            ← dataset horario validado (5.993 h, días 25–31 ausentes)
  build_tables.py       ← genera params.json desde hourly.csv
  eval_windows.py       ← evaluador pseudo-test (días 18–24) — oráculo local
  test_robustez.py      ← 18 casos límite + secuencia 72h (ejecutar antes de cada envío)
  blindspot.py          ← análisis de picos fuera de horizonte (evidencia NO-GO)
  retest48.py           ← re-test limpio horizonte 48h (A=B=C)
  replica_72h.py        ← réplica del dashboard oficial
  sweep48.py / sweep_reserve.py / sweep_gate.py  ← NO-GO horizonte (Pei)
  train_lgbm.py / lgbm_agent.py / eval_lgbm.py   ← NO-GO forecast (Pei)
  master_dataset.csv / build_dataset.py          ← dataset rico (Jan)
  params_rich.json / build_tables_rich.py / mpc_agent_rich.py  ← tablas 12×24×7
best_v1/                ← agente congelado (6.43) — NO TOCAR
submission/             ← agent.py + params.json listos para zip
submission.zip          ← empaquetado actual
SUBMISSIONS.md          ← log completo de envíos + experimentos (Elayne)
PROGRESS.md             ← estado técnico compartido
PROGRESS_jan.md         ← notas de datos/features (Jan)
```
