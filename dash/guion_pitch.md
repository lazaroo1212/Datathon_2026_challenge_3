# Guion definitivo del pitch — Battery Ops Cockpit (v3)
**Equipo Yukiyuk · 4 estudiantes de Bioinformática (UPC) · Duración: 3 min 30 s**
Modo de uso: cada bloque indica tiempo, pestaña del dashboard, texto literal y qué
mostrar/clicar. El bloque 0 es el checklist técnico previo. El bloque 7 prepara el Q&A.

---

## Bloque 0 — Checklist técnico previo (2 min, antes de que entre el jurado)

1. `git pull` en el portátil de la demo.
2. Arrancar: `.venv/bin/streamlit run dash/app.py` → se abre `localhost:8501`.
3. **Calentar la caché**: recorrer las 6 pestañas en orden (2–4 s cada una la primera
   vez; después instantáneo).
4. Pestaña 1 abierta y lista, slider de capacidad en **2.000 kWh**.
5. Cerrar pestañas del navegador que no sean la app · zoom 100% · pantalla limpia.
6. Plan B: carpeta `dash/capturas/` con una captura por pestaña.

---

## Bloque 1 — Apertura + pestaña 1 (0:00–0:55) · "¿Compramos la batería?"

**[0:00–0:25] Contexto y resultado** *(pestaña 1, KPIs grandes a la vista)*

> "Buenos días. Somos Yukiyuk: cuatro estudiantes de Bioinformática de la UPC. Nuestro
> oficio es construir pipelines de datos y no creerse ningún resultado sin un control
> adecuado — y hoy hemos aplicado exactamente esa disciplina al sistema eléctrico del
> campus. UPC Sostenible pregunta si merece la pena instalar una batería de 2 megavatios-
> hora. Nuestro resultado, verificado en la plataforma oficial: **8.957 euros y 4.276
> kilos de CO₂ ahorrados en 12 semanas** — unos 39.000 euros y 19 toneladas al año —
> con **cero infracciones físicas**. Este dashboard es la herramienta para decidir."

**[0:25–0:55] La primera decisión: ¿qué tamaño?** *(mover el slider de capacidad en vivo)*

> "Primera decisión: ¿qué tamaño de batería comprar? El slider re-simula la semana real
> con nuestro mismo agente de control. Miren la curva: pasar de 2 a 4 megavatios-hora NO
> duplica el ahorro — los rendimientos son decrecientes, porque la demanda del campus es
> veinte veces mayor que la batería. Con el coste de instalación parametrizado, el
> payback está entre 8 y 12 años. **Decisión 1: 2 MWh es el tamaño correcto; más batería
> no añade valor proporcional.**"

Mostrar: slider 500 → 4000 → volver a 2000. Señalar la línea vertical "batería propuesta".

---

## Bloque 2 — Pestaña 2 (0:55–1:20) · "¿Cómo se opera?"

> "Segunda decisión: cómo operarla. El agente planifica con optimización lineal sobre
> 24 horas: carga al mediodía —energía barata y limpia— y descarga en los picos de la
> tarde. El mapa de calor muestra dónde se gana el dinero: concentrado entre las 18 y
> las 21 horas. Y tiene forma distinta para euros y para CO₂ — porque el precio de la
> luz y el mix eléctrico no siguen el mismo reloj, y nuestro control captura los dos.
> **Decisión 2: operarla con control predictivo, no con reglas fijas.**"

Mostrar: gráfico de balance (izquierda) 5 s, SOC (derecha) 5 s, los dos heatmaps.

---

## Bloque 3 — Pestaña 3 (1:20–1:50) · "¿Priorizamos CO₂ o dinero?"

> "Tercera decisión: ¿se puede priorizar CO₂ sobre dinero? Aquí descubrimos algo
> contraintuitivo. El score oficial es 0,7 por CO₂ más 0,3 por coste. Esa fórmula define
> una **recta**: los diez primeros equipos de la clasificación caen exactamente sobre
> ella. Mover el peso entre los dos objetivos solo **desliza** el punto sobre la recta —
> el total no cambia; lo verificamos con dos envíos reales al servidor que dieron el
> mismo score. **Decisión 3: no hay un dial que girar. La única forma de salir de la
> recta es hardware: más batería o más solar.**"

Mostrar: recta gris con la estrella roja de Yukiyuk; recorrer los equipos con el ratón.

---

## Bloque 4 — Pestaña 4 (1:50–2:30) · "Hard Mode: solar ×12"

> "La organización lanzó un modo avanzado: doce veces más placas solares —con
> excedentes de energía gratis al mediodía—, una batería triple, y solo 3 horas de
> precios visibles: el resto hay que predecirlo. Nuestra progresión: 19,05 → 21,52 →
> 22,79. El salto más grande vino de **encontrar un error conceptual en nuestro propio
> optimizador**: pagaba por cargar con energía solar que en realidad era gratis. Una
> línea de código de física correcta valió **+0,68 puntos**. Y la lección negativa,
> también con datos: sustituir nuestros forecast por LightGBM con la mitad de error
> **empeoró** el score — el optimizador premia la estructura del error, no su media.
> **Decisión 4: en el escenario ampliado, la palanca real vuelve a ser hardware — más
> solar, no más software.**"

Mostrar: gráfico de barras de progresión con la línea del líder; señalar el salto v1→v2.

---

## Bloque 5 — Pestaña 6 (2:30–2:50) · "Método: por qué creerse estos números"

> "¿Por qué creerse estos números? Tres razones. Uno: el oracle — dar al control la
> información perfecta del futuro solo mejora 4 centésimas; estamos al 99% del techo.
> Dos: la transparencia — detectamos que la métrica oficial parece evaluar solo media
> semana; construimos un agente que lo confirmó en el servidor, medimos su coste oculto
> — el campus salía perjudicado — y decidimos reportar el número honesto. Tres: la
> robustez — veintiún casos límite, cero infracciones en todos los envíos, y un entorno
> local calibrado contra el servidor al céntimo."

Mostrar: las 3 métricas del caso de estudio (predicción / resultado / impacto real).

---

## Bloque 6 — Pestaña 5 (2:50–3:00) · "La evidencia, en una tabla"

> "Toda la tabla de hipótesis que han visto — abierta, descartada o confirmada — está
> en el repositorio del equipo, con los números y los scripts para reproducirla."

Mostrar: la tabla completa 3 segundos, sin leerla.

---

## Bloque 7 — Conclusiones y cierre (3:00–3:30)

> "Conclusiones para UPC Sostenible. Una: la batería de 2 MWh con nuestro control ahorra
> 39.000 euros y 19 toneladas de CO₂ al año, y el tamaño propuesto es el correcto. Dos:
> el control predictivo extrae el 99% del valor posible — la siguiente palanca de
> descarbonización es ampliar la generación solar, no el software. Y tres: estas
> decisiones no son opiniones — cada una viene de un experimento medido, y las que
> fallaron también están documentadas. Somos Yukiyuk, y esto es nuestro trabajo.
> Gracias."

*(cierre mirando al jurado; el dashboard queda en la pestaña 1)*

---

## Bloque 8 — Q&A: las 6 preguntas difíciles, preparadas

**Q1. "¿Cuál es vuestro resultado: 6,43 o 6,62?"**
> "6,43 es nuestro impacto real. El 6,62 fue un agente experimental que confirmó que la
> evaluación puntúa solo media semana; su impacto real era peor, así que lo documentamos
> como hallazgo sobre la métrica y reportamos el número honesto." *(pestaña 6)*

**Q2. "¿Y si falla vuestra predicción de consumo?"**
> "Lo medimos: el score es insensible a errores de ±25 kW, porque las decisiones las
> manda el precio y el CO₂ de las próximas horas, que el sistema conoce con exactitud."

**Q3. "¿De dónde sale el coste de la batería del payback?"**
> "Es parametrizable: 150–300 €/kWh es el rango típico de sistemas industriales
> instalados. El slider permite recalcular el payback con cualquier cifra de su proveedor."

**Q4. "¿Por qué no usáis un modelo de ML más potente para predecir?"**
> "Lo probamos, tres veces: mejoraba el error medio y empeoraba el score — el
> optimizador necesita estructura, no precisión media. Está documentado con los
> experimentos en el repositorio." *(pestaña 4, lección negativa)*

**Q5. "¿Cómo sabéis que vuestro entorno local refleja el servidor?"**
> "Calibrado al céntimo: nuestro simulador reproduce exactamente los scores oficiales
> en la misma configuración, y un envío accidental (583 infracciones, −116 puntos) nos
> enseñó dónde están los límites — por eso llevamos el clamp defensivo."

**Q6. "¿Qué aporta gente de Bioinformática a un problema de energía?"**
> "Exactamente lo que han visto: pipelines reproducibles, experimentos con control,
> y desconfianza sistemática hacia los resultados sin validación. El problema era
> distinto; el método es el mismo."

---

## Reglas de oro de la demo

- Un solo portátil maneja la app; el resto del equipo habla y vigila el tiempo.
- Caché caliente SIEMPRE (bloque 0). Nada de primera simulación en vivo.
- Si algo peta: capturas de respaldo y seguir — no depurar en directo.
- El 6.62: 30 segundos, tono neutro, terminado en la decisión del equipo. Nunca como
  logro, nunca oculto.
