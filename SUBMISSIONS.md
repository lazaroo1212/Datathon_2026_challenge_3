# Registro de envíos — Team **Yukiyuk**

**Repte**: UPC Sostenible · Optimització de la Bateria del Campus (Datathon 2026, Challenge 3)
**Última actualización**: 2026-11-14

---

## 1. Cómo se calcula el score (verificado con el envío #1)

```
Score = 0.70 · ΔCO₂% + 0.30 · ΔCost% − penalización·Infracciones
```

Check con el envío #1: `0.7 × 5.95 + 0.3 × 7.54 = 4.165 + 2.262 = 6.43 ✓`
(se redondean a 2 decimales los componentes; el desglose cuadra exacto).

- **Prioridad de palancas**: +1 pp de CO₂ vale **+0.70 pts**; +1 pp de Costo vale **+0.30 pts** → el foco de mejora debe ser CO₂ (70% del peso).
- **Infracciones**: el enunciado dice −1.00 pt c/u; con 0 infracciones aún no hemos verificado la penalización real de la plataforma. Mantener el clamping defensivo sí o sí.
- **Referencias del campus sin batería** (12 semanas de test): **71,816 kg CO₂** y **118,746.34 €**.

## 2. Cuota de envíos (10 por franja, 30 en total)

| Franja | Ventana | Usados | Restantes |
|---|---|---|---|
| Torn 1 (sábado día) | hasta 21:00 | **4**¹ | 6 |
| Torn 2 (sábado noche) | 21:00 → dom 08:00 | 0 | 10 |
| Torn 3 (domingo día) | desde 08:00 | 0 | 10 |

¹ #2 = duplicado accidental del #1. #4 (κ=0) salió idéntico al #1. Ninguno dañó el score (cuenta el mejor), pero cada uno gastó cuota.

> Total gastado: **4/30**. Cada envío restante debe tener hipótesis y métrica de éxito.

## 3. Estado actual (tras 4 envíos + ola de experimentos)

- **Mejor score: 6.43** · el último envío es también el mejor → **blindados** ante cualquier regla de ranking (mejor o último).
- **Espacio de hiperparámetros AGOTADO** (verificado en servidor y sweeps): γ ✗ · κ ✗ · β ✗ · β_mem ✗ — el MPC puntúa 6.43 con cualquier configuración de estos knobs. Agente **robusto**, no afinado por suerte.
- **Espacio estructural TAMBIÉN CERRADO (medido, ver §7)**:
  - Forecast ✗: tablas ricas (67 vs 91 kW MAE) y LightGBM (+0.002 < umbral 0.02) **no mueven el score** — con net-load ≫ batería, `g = net_f + p` nunca satura y el LP decide solo con `w` (conocido exacto) y SOC; los errores de load no cambian decisiones.
  - Horizonte 48 h ✗: retest limpio **A=B=C=5.534** → neutro, sin artefacto de cola. La cota del blindspot (2.37 pts) era límite superior que no se materializa.
- **Configuración canónica**: `submission/` (κ=0, γ=0.264). `best_v1/` congelado como referencia. No hace falta best_v2.
- **Entorno local validado como oráculo**: `replica_72h.py` reproduce el patrón del dashboard oficial (cargas +500 kW de mañana, descargas −500 kW en picos, SOC 0→100→0, arranque 50%) → `evaluate_local.py` es fiable para aprobar/rechazar envíos.
- El tier **6.44 empieza en 6.437 reales** (+0.009): hoy **no hay palanca conocida** que lo mueva. El gap local→plataforma (6.12→6.43) es de período de datos, no de método.
- **Techo estimado**: oracle con foresight perfecto ≈ 6.16 local → **~6.47 en plataforma** (offset +0.31 observado).

**Lo que queda por hacer**

1. **Robustez domingo** (guards NaN, timestamps raros, fallback si `linprog` falla) — protege, no suma.
2. Higiene de envíos: no subir nada que no bata 6.12 local en ≥0.02 y 0 infracciones (criterio `evaluate_local.py` + `SUBMISSIONS.md`).
3. Cualquier idea nueva pasa primero por el filtro: ¿es γ/κ/forecast/horizonte? → entonces ya sabemos que da 0. Buscar palancas de otra naturaleza o aceptar el 6.43 y blindar.

**Palancas cerradas (ola de experimentos 14/11, detalle en §7)**

- ~~Horizonte >24 h~~ — **NO-GO (experimento Pei)**: probados 5 mecanismos sobre el LP, todos ≤ 0 vs baseline 6.118: (a) extensión w con perfil mensual −0.43, (b) mezcla con persistencia −0.07…−0.43, (c) reserva SOC fija por la tarde −0.18…−0.96, (d) reserva SOC *gated* (solo si pico lejano > ratio·pico visible, 27 configs) −0.01…−0.52, (e) valor terminal basado en pico lejano −0.32…−1.30. **Por qué**: la batería cicla ~2000 kWh/día (throughput limitante); guardar para el pico de mañana cuesta igual valor hoy (picos diarios ≈ mismos w) y bloquea la recarga de mediodía. El bound de blindspot.py (2.37 pts) asumía 1000 kWh recolocables por episodio — en la práctica el LP ya descarga lo que vale la pena dentro de la ventana y la recolocación neta es ~0. Con el oracle a +0.04, el MPC de 24 h está en su techo estructural. Scripts: `work/sweep48.py`, `work/sweep_reserve.py`, `work/sweep_gate.py`. Agente revertido a la config congelada (h_extra=0, reserve=0) — verificado 6.118 / 0 infracciones.
- ~~Forecast más fino del net-load~~ — **DESCARTADO (experimento Pei)**: LightGBM directo t+1..t+24 (20 features observables, in_hidden_week==0, pesos imputed_frac) da **+0.002 local** vs tablas+memoria (umbral +0.02). Ver `work/train_lgbm.py`, `work/lgbm_agent.py`, `work/eval_lgbm.py`.
- ~~Revisar patrón de ciclos: ¿el corte a 24 h deja valor sobre la mesa?~~ → **respondido: no** (ver NO-GO horizonte).

## 4. Log de envíos

| # | Cuándo | Agente (γ) | **Score** | ΔCO₂ | ΔCosto | Infr. | Conclusión |
|---|---|---|---|---|---|---|---|
| — | — | Tutorial plataforma (—) | 2.16 | ? | ? | ? | punto de partida |
| 1 | sab 14:13 | MPC v1 (0.264) | **6.43** | 5.95% | 7.54% | 0 | mejor actual · proxy local 6.12 · 21è |
| 2 | sab ~14:1x | duplicado accidental del #1 | 6.43 | = | = | 0 | solo gasta cuota |
| 3 | sab tarde | MPC v2 (0.259) | **6.43** | 5.96% | 7.52% | 0 | +0.0001 reales → **afinar γ no mueve el score** |
| 4 | sab tarde | MPC v1 κ=0 (0.264) | **6.43** | 5.95% | 7.54% | 0 | idéntico al gramo al #1 (4,276.3 kg / 8,957.45 €) → **κ tampoco mueve nada** |

*Datos exactos por si hacen falta: #1 = 4,276.3 kg / 8,957.45 € · #3 = 4,282.3 kg / 8,934.47 € · líder del tier 6.44 ≈ 5.99% / 7.48% = 6.437 reales.*

## 5. Detalle por envío

### Envío #1 — 2026-11-14 14:13:26 — MPC v1 (best_v1 congelado)

- **Zip**: `agent.py` (MPC-LP, `work/mpc_agent.py`) + `params.json` (tablas de perfil). Validado con `check_submission.py` antes de subir.
- **Dry-run** (no consume cuota):
  - 48 h verificadas · init 0.6 ms · **1.698 ms/paso** → ~3.4 s de 60 s (margen 17×)
  - 0 infracciones · consignas `[112.57, 499.98, 0.0, 0.0, 0.02, 0.0] kW` coherentes
  - Fitxers auxiliares detectados: `['params.json']`
- **Resultado oficial**: Score **6.43** · CO₂ −5.95% (4,276.3 kg) · Factura −7.54% (8,957.45 €) · 0 infracciones.
- **Learnings**:
  - El entorno del servidor replica la física local (misma interfaz, mismo clamping) → el workflow local (eval_windows + sweep) es fiable para iterar.
  - Proxy local 6.12 vs oficial 6.43: el test oculto (días 25–31) es ligeramente más favorable que nuestras pseudo-ventanas (18–24). Puede ser efecto temporal (Patrones de días concretos), no sesgo de método.
  - Estamos en el paquete líder: para despegar hacen falta ganancias estructurales (horizonte largo, estacionalidad), no micro-ajustes.

### Envío #3 — MPC v2 (γ 0.264 → 0.259)

- **Hipótesis**: bajar γ desliza el reparto hacia el punto del líder (más CO₂, menos costo) con el mismo total.
- **Resultado**: 6.43 (5.96% CO₂ / 7.52% costo · 4,282.3 kg · 8,934.47 €) · 0 infracciones.
- **Conclusión**: +0.0001 pts reales. El γ teórico era exacto; **no enviar más versiones de γ**. Siguiente palanca: estructura (horizonte >24 h).

### Envío #4 — MPC v1 con κ=0 (γ 0.264)

- **Hipótesis**: κ 0.5→0.0 podría cambiar decisiones de final de horizonte.
- **Resultado**: 6.43, **idéntico al gramo al #1** (4,276.3 kg / 8,957.45 €). La trayectoria en el servidor no cambió.
- **Conclusión**: κ muerto. Con γ y κ fuera, **el espacio de hiperparámetros está agotado** — las siguientes palancas son estructurales (horizonte 48 h, forecast ML).

### Plantilla para próximos envíos (copiar y rellenar)

```
### Envío #N — AAAA-MM-DD HH:MM — <nombre agente>
- Zip: <archivos incluidos>
- Hipótesis: <qué cambiamos y por qué esperamos X pts>
- Dry-run: <pasos / latencia / infracciones / consignas>
- Resultado oficial: Score _ · CO₂ _% · Factura _% · Infr. _
- Comparativa: vs #N-1 (_) → +/− _ pts
- Learnings: <qué aprendemos / qué descartamos>
- Decisión: <congelar | iterar | descartar línea>
```

## 6. Anexo — Snapshot leaderboard (2026-11-14, ventana visible 12è–21è)

| Pos. | Registro | Equipo | Score | CO₂ | Costo | CO₂ evitado | Ahorro | Infr. |
|---|---|---|---|---|---|---|---|---|
| 12è | 12:58:19 | Dos enginyers i mig | 6.43 | 5.95% | 7.54% | 4276.6 kg | 8957.61 € | 0 |
| 13è | 13:02:08 | Psycode | 6.43 | 5.97% | 7.50% | 4288.7 kg | 8906.02 € | 0 |
| 14è | 13:06:32 | SanHackobos | 6.43 | 5.97% | 7.51% | 4288.0 kg | 8913.11 € | 0 |
| 15è | 13:06:54 | programi qui programi | 6.43 | 5.96% | 7.52% | 4282.7 kg | 8928.54 € | 0 |
| 16è | 13:07:22 | Viquipèdia Racers | 6.43 | 5.97% | 7.50% | 4289.4 kg | 8907.87 € | 0 |
| 17è | 13:16:10 | AAA | 6.43 | 5.97% | 7.50% | 4289.3 kg | 8907.60 € | 0 |
| 18è | 13:20:49 | La MasIA | 6.43 | 5.95% | 7.54% | 4276.6 kg | 8957.61 € | 0 |
| 19è | 13:24:41 | Manguihackers | 6.43 | 5.97% | 7.50% | 4289.4 kg | 8907.87 € | 0 |
| 20è | 14:06:26 | BOLETSIA.CAT | 6.43 | 5.97% | 7.50% | 4289.4 kg | 8907.87 € | 0 |
| 21è | 14:13:26 | **Yukiyuk** | **6.43** | 5.95% | 7.54% | 4276.3 kg | 8957.45 € | 0 |

> Diagnóstico del anexo: meseta perfecta 6.43 en todo el bloque visible. Desempates internos dan preferencia al que **antes se registró** → cada punto real de mejora ahora vale más que la hora de registro.

## 7. Experimentos estructurales (sin coste de envío) — 14/11 tarde

| Experimento | Quién | Resultado | Decisión |
|---|---|---|---|
| **Blindspot del horizonte** (`work/blindspot.py`) | David | 35 episodios/año donde hay pico de `w` fuera del horizonte de 24 h; cota **superior** 2.37 pts/año | GO al experimento 48 h |
| **Tablas ricas mes×hora×dow(7)** (`work/build_tables_rich.py` → `params_rich.json`) | Jan | MAE net **67 vs 91 kW** de las tablas viejas; solo datos reales, pesos por fracción imputada, fallbacks para Nov25/Ago26. **Mismo score que tablas viejas (6.12)** | Mejor base de datos, pero **no es palanca de score** |
| **LightGBM t+1..t+24** (`work/train_lgbm.py`, `lgbm_agent.py`, `eval_lgbm.py`) | Pei | **+0.002 local** vs tablas+memoria (umbral de envío: +0.02) | **Descartado**. El LP no es sensible al error de forecast: decide con `w` (exacto) y SOC |
| **Factorial 48 h + prof_w** (días 18–24) | Jan | **−0.59** (5.53) — contaminado: el eval local trunca `preus_properes_24h` al final de ventana (`environment.py:68`) y el LP retiene SOC para horas "caras" fabricadas | No adoptar tal cual |
| **Retest 48 h limpio** (`work/retest48.py`: corre sobre días reales 11–20, puntúa 11–17; A=tablas viejas, B=ricas, C=ricas+48h) | David | **A = B = C = 5.534** → el horizonte 48 h es **neutro**, sin artefacto de cola (el −0.59 de Jan era puro truncado del eval) | **Palanca horizonte: cerrada** |
| **Réplica 72 h del dashboard oficial** (`work/replica_72h.py`, ventana Sep 18–20) | David | Patrón cualitativo idéntico al servidor: cargas ~+500 kW de mañana, descargas −500 kW en picos vespertinos, ciclos SOC 0→100→0, arranque 50% · score 5.87 en la ventana | **Entorno local validado como oráculo** del servidor → `evaluate_local.py` autorizado como juez de envíos |

**Síntesis de la ola**: el score del MPC está limitado por la **estructura del problema** (batería 2 MWh vs carga ~29 MWh/día), no por forecast, horizonte ni hiperparámetros. Las 6 palancas medidas dan 0. Lo único pendiente con valor esperado > 0 es robustez y no romper nada.

## 8. HARD MODE (pestaña independiente, cuota propia: 10+10+10)

**Qué cambia**: solar ×12, batería 6.000 kWh / ±1.500 kW, obs con forecast de solo 3 h
( el resto se predice), meteo en `noves_dades_5min` (irradiancia, temps ambient/panel,
vent) y `cap_max_kwh`/`power_max_kw` en el obs. Ranking público sobre 1.008 h (mitad
del test — mismo patrón de máscara que el estándar; NO se explota, decisión de equipo).
Baselines implícitos del servidor: ≈60.248 kg CO₂ / ≈94.988 € → γ = 0.272.

### Log de envíos Hard

| # | Cuándo | Agente | **Score** | ΔCO₂ | ΔCost | Infr. | Notas |
|---|---|---|---|---|---|---|---|
| H1 | sab ~18:35 | v0 (MPC estándar re-constante, PV×12, horizonte perfil) | 19.05 | 17.88% | 21.78% | 0 | primer contacto; envío accidental al estándar dio 583 infracciones (−116.6) — lección: clamps duros |
| H2 | sab tarde | v1 (forecasts ridge aprendidos: w, load, solar) | **21.52** | 19.89% | 25.33% | 0 | ΔCO₂ −3pp vs líder (22.94) |
| H3 | dom ~10:00 | **v2 (fix excedente: net_f sin recortar a 0 → cargar con solar sobrante es gratis, como en la física real)** | **22.2** | 20.59% | 25.96% | 0 | **+0.68 sobre v1, mayor salto honesto del datathon** |

### Experimentos Hard (sin coste de envío)

| Experimento | Resultado | Decisión |
|---|---|---|
| Oracle local (foresight perfecto, net sin recortar) | ≈20.05 proxy (pre-fix) | tras el fix el techo local sube; el top 24.36 ≈ bound recalibrado → el hueco restante es forecast |
| LGBM solar directo multi-horizonte (train honesto días 1–17) | MAE directo 83 kW (vs 156 del ridge) PERO simulación **17.65 (−0.34)** | **NO-GO**: desfase train/servido en lags + el score concentra sensibilidad en transiciones crepúsculo |
| Modelos separados CO₂/precio (pre-fix) | w-MAE 0.0173→0.0166, score sin mejora | re-probar SOBRE v2 (el fix puede cambiar la sensibilidad) |
| γ=0.272 (del baseline implícito) | ya en v1 | correcto |

### Lecciones Hard

1. **El fix del excedente**: v1 recortaba `net_f` a 0 → el LP pagaba `w·p_c` por cargar
   con solar sobrante que en la realidad es gratis (export = 0 €, 0 kg). Corrección:
   `net_f = lf − pf` sin clip; el LP con `g ≥ net_f + p_c − p_d` y `g ≥ 0` ya gestiona
   gratis el excedente. +0.68 en servidor con UNA línea de código.
2. **Clamp oficial desde obs**: `cap_max_kwh` / `power_max_kw` (con fallback a constantes).
3. La meteo en LGBM ayudó al MAE (+3–7%) pero no al score — mismo patrón que estándar.
4. Envío accidental al estándar con agente Hard: 583 infracciones, −116.6 pts — el
   contador de cuota es independiente pero el daño reputacional no; verificar siempre
   la pestaña antes de Trametre.
