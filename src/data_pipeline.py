"""
Data Pipeline - Portfolio ML Research
=====================================

Descarga y prepara datos históricos de Yahoo Finance
para el proyecto de optimización de portafolios.

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
    Datos procesados
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

# Activos iniciales del proyecto
TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # Nvidia
    "AMZN",   # Amazon
    "GOOGL",  # Alphabet
    "META",   # Meta
    "TSLA",   # Tesla
    "JPM",    # JPMorgan
    "V",      # Visa
    "MA",     # Mastercard
    "WMT",    # Walmart
    "KO",     # Coca-Cola
    "PEP",    # PepsiCo
    "XOM",    # Exxon Mobil
    "JNJ",    # Johnson & Johnson
    "PG",     # Procter & Gamble
    "CAT",    # Caterpillar
    "HD",     # Home Depot
    "NFLX",   # Netflix
    "ADBE",   # Adobe
]

# Benchmark
BENCHMARK = "^GSPC"   # S&P 500

# Fecha inicial y final
START_DATE = "2018-01-01"
END_DATE = None       # None = hasta la fecha más reciente disponible

# Directorios
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

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

    # yfinance devuelve columnas multinivel cuando
    # descargamos múltiples activos.
    adjusted = data["Adj Close"].copy()

    # Ordenar cronológicamente
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

    # Orden temporal
    prices = prices.sort_index()

    # Eliminar fechas duplicadas
    prices = prices[~prices.index.duplicated(keep="first")]

    # Convertir valores a numérico
    prices = prices.apply(pd.to_numeric, errors="coerce")

    print("\nValores faltantes antes de limpiar:")
    print(prices.isna().sum())

    # No rellenamos automáticamente con forward-fill todavía.
    # Primero queremos conocer la calidad real de los datos.
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

    # Retorno promedio diario
    statistics["mean_daily_return"] = returns.mean()

    # Volatilidad diaria
    statistics["daily_volatility"] = returns.std()

    # Retorno anualizado
    statistics["annualized_return"] = returns.mean() * 252

    # Volatilidad anualizada
    statistics["annualized_volatility"] = returns.std() * np.sqrt(252)

    # Retorno acumulado
    statistics["cumulative_return"] = (
        prices.iloc[-1] / prices.iloc[0]
    ) - 1

    # Número de observaciones
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
    quality
):
    """
    Guarda los diferentes productos del pipeline.
    """

    raw_path = RAW_DIR / "yahoo_raw.csv"
    prices_path = PROCESSED_DIR / "adjusted_prices.csv"
    returns_path = PROCESSED_DIR / "daily_returns.csv"
    statistics_path = PROCESSED_DIR / "statistics.csv"
    quality_path = PROCESSED_DIR / "data_quality.csv"

    raw_data.to_csv(raw_path)

    prices.to_csv(prices_path)

    returns.to_csv(returns_path)

    statistics.to_csv(statistics_path)

    quality.to_csv(quality_path)

    print("\nArchivos guardados:")
    print(f"  {raw_path}")
    print(f"  {prices_path}")
    print(f"  {returns_path}")
    print(f"  {statistics_path}")
    print(f"  {quality_path}")


# ============================================================
# 9. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Agregar benchmark
    # --------------------------------------------------------

    all_tickers = TICKERS + [BENCHMARK]

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
        quality
    )

    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PIPELINE TERMINADO")
    print("=" * 60)

    print("\nDimensiones de precios:")
    print(prices.shape)

    print("\nPrimeras observaciones:")
    print(prices.head())

    print("\nEstadísticas:")
    print(statistics.round(4))


if __name__ == "__main__":
    main()