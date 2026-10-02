"""
Data Pipeline Diversificado - Portfolio ML Research
====================================================

Variante del pipeline original, con un universo de activos
seleccionado por sector (GICS) en vez de estar concentrado en
tecnológicas de gran capitalización.

Objetivo: comparar el portafolio "concentrado" original contra
uno "equilibrado" sectorialmente, para evaluar si la ventaja de
Markowitz/Equal Weight sobre el S&P 500 observada en el proyecto
original depende del sesgo de selección del universo.

Misma estructura y funciones que data_pipeline.py — solo cambia
la lista de activos y las carpetas de salida (no se sobreescriben
los datos originales).

Flujo:
    Yahoo Finance
        ↓
    Datos históricos
        ↓
    Limpieza
        ↓
    Precios ajustados
        ↓
    Retornos
        ↓
    Estadísticas
        ↓
    Datos procesados (diversificados)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

# Universo diversificado: 4 empresas representativas por cada
# uno de los 11 sectores GICS (large-caps reconocidas).
# Ajusta libremente los tickers si quieres otras representativas
# de cada sector — lo importante para el experimento es que
# ningún sector esté sobrerrepresentado.
SECTORS = {
    "Tecnología":              ["AAPL", "MSFT", "NVDA", "AVGO"],
    "Salud":                   ["LLY", "UNH", "JNJ", "ABBV"],
    "Financiero":              ["JPM", "V", "MA", "BAC"],
    "Consumo Discrecional":    ["AMZN", "TSLA", "HD", "MCD"],
    "Servicios de Comunicación": ["GOOGL", "META", "NFLX", "DIS"],
    "Consumo Básico":          ["WMT", "PG", "KO", "PEP"],
    "Industrial":              ["CAT", "GE", "HON", "UNP"],
    "Energía":                 ["XOM", "CVX", "COP", "SLB"],
    "Servicios Públicos":      ["NEE", "DUK", "SO", "D"],
    "Bienes Raíces":           ["PLD", "AMT", "EQIX", "SPG"],
    "Materiales":              ["LIN", "SHW", "APD", "ECL"],
}

# Aplana el diccionario a una sola lista de tickers, y guarda
# también el mapeo ticker -> sector (útil más adelante para
# analizar composición sectorial de los pesos óptimos).
TICKERS = [ticker for sector_tickers in SECTORS.values() for ticker in sector_tickers]

SECTOR_MAP = pd.DataFrame(
    [(ticker, sector) for sector, tickers in SECTORS.items() for ticker in tickers],
    columns=["ticker", "sector"]
)

# Benchmark
BENCHMARK = "^GSPC"   # S&P 500

# Fecha inicial y final — mismas que el pipeline original, para
# que la comparación final cubra exactamente el mismo periodo.
START_DATE = "2018-01-01"
END_DATE = None       # None = hasta la fecha más reciente disponible

# Directorios — SEPARADOS del pipeline original, para no
# sobreescribir los datos del portafolio concentrado.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw_diversified"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_diversified"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DESCARGAR DATOS
# ============================================================

def download_data(tickers, start_date, end_date=None):
    """
    Descarga datos históricos desde Yahoo Finance.

    Parameters
    ----------
    tickers : list
        Lista de símbolos de Yahoo Finance.
    start_date : str
        Fecha inicial YYYY-MM-DD.
    end_date : str or None
        Fecha final.

    Returns
    -------
    pd.DataFrame
        Datos históricos descargados.
    """

    print("\nDescargando datos desde Yahoo Finance...")
    print(f"Activos: {len(tickers)}")
    print(f"Desde: {start_date}")
    print(f"Hasta: {end_date if end_date else 'última fecha disponible'}")

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=True,
        threads=True,
        group_by="column"
    )

    if data.empty:
        raise ValueError("No se descargaron datos.")

    return data


# ============================================================
# 3. EXTRAER PRECIOS AJUSTADOS
# ============================================================

def get_adjusted_prices(data):
    """
    Extrae los precios ajustados de todos los activos.
    """

    adjusted = data["Adj Close"].copy()
    adjusted = adjusted.sort_index()

    return adjusted


# ============================================================
# 4. LIMPIEZA
# ============================================================

def clean_prices(prices):
    """
    Limpia la matriz de precios.

    - Ordena fechas
    - Elimina duplicados
    - Convierte a numérico
    - Reporta valores faltantes
    """

    prices = prices.copy()

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.apply(pd.to_numeric, errors="coerce")

    print("\nValores faltantes antes de limpiar:")
    print(prices.isna().sum())

    prices = prices.dropna(how="all")

    return prices


# ============================================================
# 5. CALCULAR RETORNOS
# ============================================================

def calculate_returns(prices):
    """
    Calcula retornos simples diarios.
    """

    returns = prices.pct_change(fill_method=None)

    return returns


# ============================================================
# 6. ESTADÍSTICAS BÁSICAS
# ============================================================

def calculate_statistics(prices, returns):
    """
    Calcula estadísticas descriptivas de los activos.
    """

    statistics = pd.DataFrame(index=prices.columns)

    statistics["mean_daily_return"] = returns.mean()
    statistics["daily_volatility"] = returns.std()
    statistics["annualized_return"] = returns.mean() * 252
    statistics["annualized_volatility"] = returns.std() * np.sqrt(252)
    statistics["cumulative_return"] = (prices.iloc[-1] / prices.iloc[0]) - 1
    statistics["observations"] = returns.count()

    return statistics


# ============================================================
# 7. REPORTE DE CALIDAD
# ============================================================

def quality_report(prices, returns):
    """
    Genera un reporte sencillo de calidad de datos.
    """

    report = pd.DataFrame(index=prices.columns)

    report["missing_prices"] = prices.isna().sum()
    report["missing_returns"] = returns.isna().sum()

    report["first_date"] = [
        prices[col].first_valid_index()
        for col in prices.columns
    ]

    report["last_date"] = [
        prices[col].last_valid_index()
        for col in prices.columns
    ]

    return report


# ============================================================
# 8. GUARDAR DATOS
# ============================================================

def save_data(
    raw_data,
    prices,
    returns,
    statistics,
    quality,
    sector_map
):
    """
    Guarda los diferentes productos del pipeline, incluyendo
    el mapeo ticker -> sector (no existía en el pipeline original).
    """

    raw_path = RAW_DIR / "yahoo_raw.csv"
    prices_path = PROCESSED_DIR / "adjusted_prices.csv"
    returns_path = PROCESSED_DIR / "daily_returns.csv"
    statistics_path = PROCESSED_DIR / "statistics.csv"
    quality_path = PROCESSED_DIR / "data_quality.csv"
    sector_map_path = PROCESSED_DIR / "sector_map.csv"

    raw_data.to_csv(raw_path)
    prices.to_csv(prices_path)
    returns.to_csv(returns_path)
    statistics.to_csv(statistics_path)
    quality.to_csv(quality_path)
    sector_map.to_csv(sector_map_path, index=False)

    print("\nArchivos guardados:")
    print(f"  {raw_path}")
    print(f"  {prices_path}")
    print(f"  {returns_path}")
    print(f"  {statistics_path}")
    print(f"  {quality_path}")
    print(f"  {sector_map_path}")


# ============================================================
# 9. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Agregar benchmark
    # --------------------------------------------------------

    all_tickers = TICKERS + [BENCHMARK]

    print("\nUniverso diversificado por sector:")
    for sector, tickers in SECTORS.items():
        print(f"  {sector}: {', '.join(tickers)}")
    print(f"\nTotal de activos (sin benchmark): {len(TICKERS)}")

    # --------------------------------------------------------
    # Descargar
    # --------------------------------------------------------

    raw_data = download_data(
        all_tickers,
        START_DATE,
        END_DATE
    )

    # --------------------------------------------------------
    # Precios ajustados
    # --------------------------------------------------------

    prices = get_adjusted_prices(raw_data)

    # --------------------------------------------------------
    # Limpieza
    # --------------------------------------------------------

    prices = clean_prices(prices)

    # --------------------------------------------------------
    # Retornos
    # --------------------------------------------------------

    returns = calculate_returns(prices)

    # --------------------------------------------------------
    # Estadísticas
    # --------------------------------------------------------

    statistics = calculate_statistics(
        prices,
        returns
    )

    # --------------------------------------------------------
    # Calidad
    # --------------------------------------------------------

    quality = quality_report(
        prices,
        returns
    )

    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------

    save_data(
        raw_data,
        prices,
        returns,
        statistics,
        quality,
        SECTOR_MAP
    )

    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PIPELINE DIVERSIFICADO TERMINADO")
    print("=" * 60)

    print("\nDimensiones de precios:")
    print(prices.shape)

    print("\nPrimeras observaciones:")
    print(prices.head())

    print("\nEstadísticas:")
    print(statistics.round(4))


if __name__ == "__main__":
    main()
