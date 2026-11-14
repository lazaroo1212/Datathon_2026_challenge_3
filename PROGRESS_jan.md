# PROGRESS_JAN — notas de Jan (datos, features y experimentos)

Log personal; el estado compartido del equipo vive en `PROGRESS.md`.

## 1. Master dataset (HECHO)

- `work/build_dataset.py` → `work/master_dataset.csv` (8761 h × 63 cols) y
  `work/base_5min.csv.gz`: resample 5min→1h, load=clip(−export,0,5000),
  PV=suma 3 inversores, imputación (interp ≤15 min, medias estacionales
  jerárquicas resto), flags `*_imputed_frac`, `in_hidden_week`, festivos,
  lags 24/48/168h, rolling means past-only, meteo. 0 huecos internos.
- Trampa cubierta: la franja 17 ago–1 sep 2026 está imputada pero NO marcada
  como `in_hidden_week` (el flag solo marca días >24) → filtrar además por
  `*_imputed_frac`.

## 2. Tablas ricas mes×hora×dow (HECHO)

- `work/build_tables_rich.py` → `work/params_rich.json`: tablas (12, 24, 7)
  de net/pv/w **solo datos reales** (5993 h, peso = (1−imput_load)·(1−imput_pv)).
  Mismo formato que `params.json` (+ `axis3`, `cnt_net`).
- MAE net 67 vs 91 kW de las tablas wknd (pseudo-test días 18–24: 60.3 vs 80.4).
- Nov 2025 no existe en los crudos (168 celdas vacías → jerarquía de fallbacks
  dow → wknd → hora). Aug solo días 1–16 reales.
- La ponderación por fracción real da lo mismo que dropna duro (67.1 vs 67.2).

## 3. Experimento horizonte-48h (resultado NEGATIVO — no adoptar)

Factorial 2×2 en pseudo-test (días 18–24, `eval_windows.py`):

| tablas | horizonte | score |
|---|---|---|
| viejas (12,24,2) | H=KNOWN | **6.12** (agente actual) |
| ricas (12,24,7) | H=KNOWN | 6.12 (idéntico) |
| viejas | 48h + prof_w | 5.53 |
| ricas | 48h/24h + prof_w | 5.53 |

Hallazgos:
1. **Las tablas de forecast no afectan el score**: con net-load ≫ batería,
   `g = net_f + p` nunca se satura y el LP decide solo con w (conocido) y SOC.
   Confirma a nivel de decisión el límite del oráculo (§7 de PROGRESS.md).
2. **Extender el horizonte con w estimado (prof_w) RESTA −0.59**: en el tail
   de cada ventana el LP retiene SOC para horas "caras" fabricadas que no
   existen. (El eval local trunca los arrays de precio al final de ventana —
   `environment.py:68`; la plataforma real siempre da 24h.)
3. `mpc_agent_rich.py` queda con `extend=0` por defecto (equivale al agente
   actual, consume params_rich.json). No subir a plataforma sin más pruebas.

## 4. Siguientes pasos de Jan

- Nada de forecast da puntos: el margen está en control (ver §3) y en
  calibrar contra el servidor (6.43 vs proxy 6.12).
- Ideas si sobra tiempo: w-profiles por mes para el **fallback** del greedy;
  revisar `replica_72h.py` del compañero (score 5.87 como oráculo local).
