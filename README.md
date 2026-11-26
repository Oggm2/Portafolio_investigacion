# Portafolio de Inversión Cuantitativo con Machine Learning

Proyecto de investigación en finanzas cuantitativas que evalúa si las señales de machine learning aportan valor a la construcción de un portafolio de acciones estadounidenses. El criterio de éxito no es únicamente reducir el error de predicción: una estrategia debe mejorar, de manera consistente y fuera de muestra, el rendimiento ajustado por riesgo frente a referencias simples y al S&P 500.

> **Nota metodológica.** SVR y LSTM son líneas de investigación, no modelos finales asumidos. Se mantienen sólo si superan las referencias bajo una evaluación temporal rigurosa.

## Objetivo

Responder la pregunta: ¿la incorporación de predicciones de retornos futuros mejora un portafolio optimizado respecto a Equal Weight, Markowitz tradicional y el S&P 500?

```text
Yahoo Finance → limpieza → EDA → features → Markowitz → SVR/LSTM
→ backtesting temporal → costos y robustez → comparación final
```

## Universo y datos

- Precios diarios ajustados descargados de Yahoo Finance mediante `yfinance`.
- Benchmark: S&P 500 (`^GSPC`).
- Universo principal: 44 acciones large-cap, con cuatro representantes de cada uno de los 11 sectores GICS. El mapeo está en [`data/processed_diversified/sector_map.csv`](data/processed_diversified/sector_map.csv).
- Universo original: 20 acciones, conservado para contrastar el efecto de la concentración sectorial.
- Inicio de la muestra: enero de 2018; el final se actualiza al descargar los datos.

## Metodología

### Variables y predicción

Las variables explicativas incluyen retornos acumulados a 1, 5, 10, 21 y 63 días, medias móviles y exponenciales, volatilidad móvil, RSI, MACD y bandas de Bollinger. El objetivo base es el retorno simple de los siguientes cinco días; también se explora un horizonte de 21 días.

Se comparan SVR con kernel RBF y LSTM. Ambos modelos se evalúan con divisiones cronológicas; las métricas predictivas incluyen RMSE, R² y métricas de ranking cuando corresponde. Un R² negativo no se interpreta como evidencia de señal.

### Portafolio y validación

Las estrategias incluyen Equal Weight, Markowitz de máximo Sharpe y mínima volatilidad, además de variantes que usan retornos esperados estimados por SVR/LSTM. Se imponen restricciones de no ventas en corto y límites máximos por activo.

El análisis principal utiliza ventanas temporales crecientes y rebalanceo fuera de muestra. La estimación de retornos y covarianza usa exclusivamente información disponible en cada fecha de decisión. Las evaluaciones consideran retorno anualizado, volatilidad, Sharpe, máximo drawdown, rotación y costos de transacción.

## Resultados destacados

En la evaluación fuera de muestra del universo diversificado con rebalanceo, los mejores resultados de referencia fueron obtenidos por Markowitz de máximo Sharpe con un límite de 15% por activo: Sharpe 0.964, frente a 0.911 para Equal Weight de comprar y mantener y 0.636 para el S&P 500. Los resultados detallados están en [`results/backtesting_rebalanceo_diversified_min0.01_est504_reb63/summary_oos.csv`](results/backtesting_rebalanceo_diversified_min0.01_est504_reb63/summary_oos.csv).

Los experimentos por ventanas muestran que SVR y LSTM no han demostrado una ventaja estable frente a Equal Weight. Por ello, el resultado del proyecto no depende de que estos modelos sean seleccionados como estrategia final.

## Estructura del repositorio

```text
data/
├── raw/                         # Descargas del universo original
├── raw_diversified/             # Descargas del universo sectorial
├── processed/                   # Precios, retornos y features originales
└── processed_diversified/       # Datos del análisis principal
notebooks/
├── 02–04                        # EDA, features y Markowitz
├── 05–06                        # Experimentos SVR y LSTM
├── 05c–06c                      # Robustez por ventanas temporales
├── 08                           # Backtesting con rebalanceo
└── 09                           # Comparación final de estrategias
results/                         # Tablas, figuras y backtests reproducibles
scripts/
└── run_walk_forward_svr.py      # Backtest SVR sin leakage (universo original)
src/
├── data_pipeline.py             # Pipeline del universo original
├── data_pipeline_diversified.py # Pipeline del universo principal
└── walk_forward_backtest.py     # Utilidades de validación walk-forward
```

## Instalación

Se recomienda Python 3.10 o posterior y un entorno virtual aislado.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib yfinance ta scikit-learn tensorflow PyPortfolioOpt jupyter
```

> En Windows, TensorFlow puede requerir una configuración particular según la versión de Python y el hardware. El resto del flujo puede ejecutarse sin LSTM.

## Cómo reproducir el análisis principal

1. Descarga y prepara el universo diversificado:

   ```powershell
   python src/data_pipeline_diversified.py
   ```

2. Abre Jupyter y ejecuta los notebooks en este orden:

   ```powershell
   jupyter notebook
   ```

   - `02_EDA_diversified.ipynb`
   - `03_feature_engineering_diversified.ipynb`
   - `04_markowitz_diversified.ipynb`
   - `05_svr_diversified.ipynb`
   - `06_lstm_diversified.ipynb`
   - `05c_svr_ventanas.ipynb` y `06c_lstm_ventanas.ipynb`
   - `08_backtesting_rebalanceo_diversified.ipynb`
   - `09_comparacion_definitiva.ipynb`

3. Revisa las tablas finales en `results/` y las figuras de comparación de estrategias en `results/comparacion_definitiva/`.

## Backtest SVR walk-forward

El siguiente script implementa una evaluación adicional para el universo original: purga etiquetas aún no realizadas, ajusta escaladores dentro de cada pliegue temporal, recalcula la covarianza con datos pasados y descuenta costos por rotación.

```powershell
python scripts/run_walk_forward_svr.py --oos-start 2024-01-01 --cost-bps 10
```

Para un horizonte de 21 días:

```powershell
python scripts/run_walk_forward_svr.py --horizon 21 --oos-start 2024-01-01 --cost-bps 10
```

Los archivos se guardan en `results/walk_forward_svr_h5/` o `results/walk_forward_svr_h21/`. Los notebooks `05_svr.ipynb` y `06_lstm.ipynb` son exploratorios; sus pesos estáticos no deben presentarse como evidencia fuera de muestra.

## Limitaciones

- Los datos históricos no garantizan resultados futuros.
- Yahoo Finance puede modificar, corregir o limitar series históricas.
- Los costos modelados son aproximaciones; no sustituyen comisiones, spread, liquidez, impuestos ni impacto de mercado reales.
- El proyecto es académico y no constituye asesoría de inversión.

## Licencia

Este proyecto se distribuye bajo la licencia incluida en [LICENSE](LICENSE).
