"""Run the leakage-free SVR experiment from the repository root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.walk_forward_backtest import WalkForwardConfig, save_results, walk_forward_svr

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d", "return_21d", "return_63d",
    "sma_10", "sma_20", "sma_50", "ema_10", "ema_20", "ema_50",
    "vol_10", "vol_20", "vol_50", "rsi_14", "macd", "macd_signal",
    "bb_high", "bb_low",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos-start", default="2024-01-01")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--refit-every", type=int, default=21)
    args = parser.parse_args()

    processed = PROJECT_ROOT / "data" / "processed"
    features_name = "features_dataset.csv" if args.horizon == 5 else f"features_dataset_h{args.horizon}.csv"
    features = pd.read_csv(processed / features_name, parse_dates=["Date"])
    returns = pd.read_csv(processed / "daily_returns.csv", index_col=0, parse_dates=True)
    config = WalkForwardConfig(
        horizon=args.horizon,
        transaction_cost_bps=args.cost_bps,
        refit_every=args.refit_every,
    )
    backtest, weights, predictions, metrics = walk_forward_svr(
        features=features,
        daily_returns=returns,
        feature_columns=FEATURE_COLUMNS,
        target_column=f"target_return_{args.horizon}d",
        benchmark="^GSPC",
        out_of_sample_start=args.oos_start,
        config=config,
    )
    output = PROJECT_ROOT / f"results/walk_forward_svr_h{args.horizon}"
    save_results(output, backtest, weights, predictions, metrics)
    print(metrics.round(4))
    print(f"Resultados guardados en: {output}")


if __name__ == "__main__":
    main()
