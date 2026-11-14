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

## 3. Estado actual (tras 4 envíos)

- **Mejor score: 6.43** · el último envío es también el mejor → **blindados** ante cualquier regla de ranking (mejor o último).
- **Espacio de hiperparámetros AGOTADO** (verificado en servidor y sweeps): γ ✗ · κ ✗ · β ✗ · β_mem ✗ — el MPC puntúa 6.43 con cualquier configuración de estos knobs. Agente **robusto**, no afinado por suerte.
- **Configuración canónica**: `submission/` (κ=0, γ=0.264). `best_v1/` congelado como referencia. No hace falta best_v2.
- El tier **6.44 empieza en 6.437 reales** (+0.009). Solo se alcanza con **estructura**, no parámetros.
- **Techo estimado**: oracle con foresight perfecto ≈ 6.16 local → **~6.47 en plataforma** (offset +0.31 observado). Ahí es donde apuntamos.

**Ideas en cola (prioridad)**

1. **Horizonte >24 h** con perfil mensual de `w` — el LP termina ciego cada medianoche.
2. ~~Forecast más fino del net-load~~ — **DESCARTADO (experimento Pei, 14/11)**: LightGBM directo t+1..t+24 (20 features observables, in_hidden_week==0, pesos imputed_frac) da **+0.002 local** vs tablas+memoria (umbral +0.02). El LP no es sensible a errores de forecast de load (±25 kW): las decisiones las manda el paisaje w[t] conocido. Ver `work/train_lgbm.py`, `work/lgbm_agent.py`, `work/eval_lgbm.py`.
1. **Horizonte 48 h** con tablas ricas de Jan (`params_rich.json`, hour × dow × month, `in_hidden_week == 0`).
2. Forecast ML de Pei (LightGBM consumo/solar t+1..t+24) — **umbral de envío: batir a tablas+memoria por ≥0.02** en `evaluate_local.py` (el techo oracle dice que no invertir más de media jornada).
3. Revisar patrón de ciclos: el SOC toca 0/100 a menudo — ¿el corte a 24 h deja valor sobre la mesa?

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
