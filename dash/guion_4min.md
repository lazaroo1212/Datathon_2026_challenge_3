# Guion 4 min — Yukiyuk (una página por persona)
**Duración objetivo: 4 minutos · 4 hablantes · demo en directo del dashboard**

---

## JAN · 0:00–1:00 · Pestaña 1 · ¿Comprar la batería?

EN PANTALLA: pestaña 1. Señalar los cuatro KPIs superiores; después la curva de
capacidad y el payback.

«Buenos días. Somos Yukiyuk: cuatro estudiantes de Bioinformática de la UPC. Nuestro
oficio es construir pipelines de datos y no creernos ningún resultado sin un control
adecuado — y hoy aplicamos exactamente esa disciplina al sistema eléctrico del campus.»

«Os presentamos Battery Ops Cockpit, un dashboard que convierte nuestro trabajo en el
Datathon en decisiones concretas para un campus más sostenible.»

«La primera decisión: ¿merece la pena instalar una batería? Con 2 MWh y 500 kW, nuestro
agente obtiene un score oficial de 6,43 con cero infracciones. En las doce semanas
evaluadas evitamos 4.276 kilos de CO₂ y ahorramos 8.957 euros — unos 39.000 euros y 19
toneladas al año.»

«Y no nos quedamos en la cifra: aquí se puede cambiar capacidad, potencia y coste de
instalación. La curva muestra rendimientos decrecientes — duplicar la batería no duplica
el ahorro — y el payback depende del coste real del sistema. Pasamos de una puntuación a
una decisión de inversión. Pei os enseña cómo se consigue ese ahorro.»

## PEI · 1:00–2:00 · Pestaña 2 · ¿Cómo opera?

EN PANTALLA: pestaña 2. Gráfico de demanda/importación, estado de carga, y bajar a los
dos mapas de calor.

«La segunda pregunta es cómo opera la batería. Usamos un control predictivo — MPC — que
decide cada hora cuándo cargar y descargar considerando demanda, precio de electricidad
y emisiones de CO₂.»

«En este gráfico comparamos la demanda neta del campus con la importación cuando actúa
la batería; al lado, su estado de carga: carga al mediodía, descarga en los picos.
Físicamente coherente.»

«Y abajo, los dos mapas de calor cuentan la mejor historia del dashboard: el ahorro
económico y la reducción de CO₂ siguen patrones casi idénticos — picos entre las 18 y
las 22 horas y otro pico matinal. No es casualidad: cuando la batería descarga y reduce
la importación, gana los dos beneficios a la vez. En el mercado español el precio y el
carbono van de la mano.»

«El mensaje para UPC Sostenible: reducir costes y descarbonizar no compiten — la misma
descarga lo consigue simultáneamente. Elayne os mostrará cómo analizamos ese equilibrio
y qué cambia en un escenario más exigente.»

## ELAYNE · 2:00–3:00 · Pestañas 3 y 4 · Trade-off y Hard Mode

EN PANTALLA: pestaña 3, señalar la recta iso-score y el barrido de gamma; después
pestaña 4 y señalar la progresión de puntuaciones.

«El score combina dos objetivos: el 70 % depende de reducir CO₂ y el 30 % de reducir
costes. Esa fórmula define una recta: los equipos que la han entendido caen exactamente
sobre ella, y mover el peso entre ambos objetivos solo desliza el punto — no mejora el
total. Lo verificamos con simulaciones y envíos reales.»

«Por eso el salto vino de otra parte: el Hard Mode. La generación solar se multiplica
por doce, la batería llega a 6 MWh y solo hay 3 horas de información futura visible.»

«Ahí encontramos un error conceptual en nuestro propio optimizador: trataba parte del
excedente solar gratuito como si tuviera coste. Corregir la representación física de ese
excedente añadió 0,68 puntos. Con la mejora siguiente llegamos a 22,79 puntos oficiales:
un 21,29 % menos de emisiones y un 26,30 % menos de coste.»

«Y probamos también lo contrario: modelos de predicción con mejor error medio que
empeoraban el score. En este problema, la estructura del error importa más que su
magnitud. David cerrará con la evidencia que respalda estas decisiones.»

## DAVID · 3:00–4:00 · Pestañas 5 y 6 · Evidencia y método

EN PANTALLA: pestaña 5, mostrar brevemente la tabla de experimentos; pasar a la pestaña
6, señalar el oracle y el experimento de evaluación; terminar mirando al jurado.

«Para nosotros el dashboard no era una colección de gráficos bonitos: registramos las
hipótesis, las pruebas y también los resultados negativos.»

«El experimento clave es el oracle: incluso dando al optimizador información futura
perfecta, el score solo mejora unas 0,04 puntos sobre el 6,43 oficial. Con esta batería,
el margen de mejora del control es muy pequeño: para reducir mucho más hay que actuar
sobre la infraestructura.»

«Y la transparencia: detectamos que una particularidad de la evaluación permitía subir
el ranking hasta 6,62, aunque el impacto energético real empeoraba. Decidimos no
presentar eso como una mejora de sostenibilidad — queda documentado, como nuestros
resultados negativos.»

«En resumen: Battery Ops Cockpit permite decidir cuánto invertir, cómo operar y qué
cambios merecen la pena, con resultados trazables y sin confundir puntuación con
impacto real. Somos Yukiyuk, cuatro de Bioinformática: el problema era distinto; el
método, el mismo. Muchas gracias.»
