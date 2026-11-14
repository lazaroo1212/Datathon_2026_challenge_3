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
| Torn 1 (sábado día) | hasta 21:00 | **3**¹ | 7 |
| Torn 2 (sábado noche) | 21:00 → dom 08:00 | 0 | 10 |
| Torn 3 (domingo día) | desde 08:00 | 0 | 10 |

¹ #2 = duplicado accidental del #1. No afecta al score (cuenta el mejor), pero gasta cuota.

> Total gastado: **3/30**. Cada envío restante debe tener hipótesis y métrica de éxito.

## 3. Estado actual (tras 3 envíos)

- **Mejor score: 6.43** (dos agentes distintos dan 6.43). El tier **6.44 empieza en 6.437 reales** → nos faltan solo **+0.009**.
- **Lección clave**: todos los 6.43 están sobre la **misma recta iso-score**. Bajar γ deslizó el reparto (+CO₂, −costo) sin mover el total (+0.0001 reales). La γ teórica `γ = 0.3·CO₂_base/(0.7·Costo_base) ≈ 0.264` es la **linealización exacta** del score. → **Eje γ: muerto, no gastar más envíos.**
- Para superar la recta hay que **desplazar la frontera** (estructura): mejores forecasts, horizonte >24 h, mejor uso del ciclo.
- **Techo estimado**: oracle con foresight perfecto ≈ 6.16 local → **~6.47 en plataforma** (offset +0.31 observado). Ahí es donde apuntamos.

**Ideas en cola (prioridad)**

1. **Horizonte >24 h** con perfil mensual de `w` — el LP termina ciego cada medianoche.
2. ~~Forecast más fino del net-load~~ — **DESCARTADO (experimento Pei, 14/11)**: LightGBM directo t+1..t+24 (20 features observables, in_hidden_week==0, pesos imputed_frac) da **+0.002 local** vs tablas+memoria (umbral +0.02). El LP no es sensible a errores de forecast de load (±25 kW): las decisiones las manda el paisaje w[t] conocido. Ver `work/train_lgbm.py`, `work/lgbm_agent.py`, `work/eval_lgbm.py`.
3. Revisar patrón de ciclos: el SOC toca 0/100 a menudo — ¿el corte a 24 h deja valor sobre la mesa?

## 4. Log de envíos

| # | Cuándo | Agente (γ) | **Score** | ΔCO₂ | ΔCosto | Infr. | Conclusión |
|---|---|---|---|---|---|---|---|
| — | — | Tutorial plataforma (—) | 2.16 | ? | ? | ? | punto de partida |
| 1 | sab 14:13 | MPC v1 (0.264) | **6.43** | 5.95% | 7.54% | 0 | mejor actual · proxy local 6.12 · 21è |
| 2 | sab ~14:1x | duplicado accidental del #1 | 6.43 | = | = | 0 | solo gasta cuota |
| 3 | sab tarde | MPC v2 (0.259) | **6.43** | 5.96% | 7.52% | 0 | +0.0001 reales → **afinar γ no mueve el score** |

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
