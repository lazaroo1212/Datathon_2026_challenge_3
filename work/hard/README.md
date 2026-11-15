# Hard Mode v1 (Yukiyuk) - paquete para el repo

- v1/            agente FINAL (entrenado con todo hourly.csv). Es el que se envia. Score servidor (publico): 21.52
- v1_honest/     mismo agent.py con params entrenados SIN los dias 18-24 de cada mes: usar este para medir mejoras locales sin fuga
- tools/sim_hard.py   simulador local Hard Mode (PV x12, 6000 kWh / 1500 kW, obs de 3 h)
- tools/train_v1.py   reentrena params.json (perfiles + ridge de forecast de w, consumo y solar x12)

El "ridge de forecast" NO es un fichero aparte: esta dentro de params.json (claves PL, PP, PW, coef).
Referencia en mi simulacion (9 ventanas concatenadas): v1_honest = 17.99, v0 = 15.86, oracle info perfecta = 20.05.
Colocar en el repo, p. ej. work/hard/ (v1/, v1_honest/, tools/).

Ejemplo:
  python3 work/hard/tools/sim_hard.py --agent work/hard/v1_honest --hourly work/hourly.csv \
      --env paquet_participants_datathon_upc/starter_kit
