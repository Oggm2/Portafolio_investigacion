# Portafolio_investigacion

## Evaluación fuera de muestra

Los notebooks `05_svr.ipynb` y `06_lstm.ipynb` son exploratorios. La
evaluación de portafolio reproducible se realiza con
`scripts/run_walk_forward_svr.py`, que purga etiquetas no realizadas, ajusta
escaladores dentro de validación temporal, calcula la covarianza sólo con datos
pasados y descuenta costos por rotación.

```powershell
python scripts/run_walk_forward_svr.py --oos-start 2024-01-01 --cost-bps 10
```

Los resultados auditables se guardan en `results/walk_forward_svr/`: retornos
diarios y acumulados, pesos por rebalanceo, predicciones y métricas.
