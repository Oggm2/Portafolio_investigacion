"""Leakage-free walk-forward utilities for portfolio experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@dataclass(frozen=True)
class WalkForwardConfig:
    """Parameters fixed before inspecting the test period."""

    horizon: int = 5
    rebalance_every: int = 5
    covariance_lookback: int = 252
    max_weight: float = 0.15
    risk_free_rate: float = 0.02
    transaction_cost_bps: float = 10.0
    n_splits: int = 5
    min_training_rows: int = 504
    refit_every: int = 21


DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "regressor__svr__C": [0.1, 1.0, 10.0],
    "regressor__svr__epsilon": [0.001, 0.01, 0.1],
    "regressor__svr__gamma": ["scale", "auto"],
}


def make_svr_search(config: WalkForwardConfig) -> GridSearchCV:
    """Create an SVR search with scaling inside each purged CV fold."""
    estimator = TransformedTargetRegressor(
        regressor=Pipeline([
            ("scale", StandardScaler()),
            ("svr", SVR(kernel="rbf")),
        ]),
        transformer=StandardScaler(),
    )
    return GridSearchCV(
        estimator=estimator,
        param_grid=DEFAULT_PARAM_GRID,
        cv=TimeSeriesSplit(n_splits=config.n_splits, gap=config.horizon),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )


def _common_feature_dates(features: pd.DataFrame, tickers: list[str]) -> pd.DatetimeIndex:
    date_sets = [set(features.loc[features["ticker"] == ticker, "Date"]) for ticker in tickers]
    return pd.DatetimeIndex(sorted(set.intersection(*date_sets)))


def _annualize_horizon_return(prediction: float, horizon: int) -> float:
    return float((1.0 + max(prediction, -0.999)) ** (252.0 / horizon) - 1.0)


def _portfolio_weights(mu: pd.Series, history: pd.DataFrame, config: WalkForwardConfig) -> pd.Series:
    """Optimize using only daily returns observable at the decision date."""
    history = history.reindex(columns=mu.index).dropna(how="any")
    if len(history) < 30:
        return pd.Series(1.0 / len(mu), index=mu.index)
    try:
        cov = risk_models.CovarianceShrinkage(history, returns_data=True).ledoit_wolf()
        frontier = EfficientFrontier(mu, cov, weight_bounds=(0, config.max_weight))
        frontier.max_sharpe(risk_free_rate=config.risk_free_rate)
        weights = pd.Series(frontier.clean_weights(), dtype=float).reindex(mu.index).fillna(0.0)
        if weights.sum() <= 0:
            raise ValueError("Optimizer returned zero total weight.")
        return weights / weights.sum()
    except (ValueError, ArithmeticError):
        return pd.Series(1.0 / len(mu), index=mu.index)


def _metrics(daily_returns: pd.Series, risk_free_rate: float) -> dict[str, float]:
    daily_returns = daily_returns.dropna()
    cumulative = (1.0 + daily_returns).cumprod()
    years = len(daily_returns) / 252.0
    cagr = cumulative.iloc[-1] ** (1.0 / years) - 1.0
    volatility = daily_returns.std(ddof=1) * np.sqrt(252.0)
    return {
        "annualized_return": cagr,
        "annualized_volatility": volatility,
        "sharpe": (cagr - risk_free_rate) / volatility if volatility else np.nan,
        "max_drawdown": (cumulative / cumulative.cummax() - 1.0).min(),
    }


def walk_forward_svr(
    features: pd.DataFrame,
    daily_returns: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    benchmark: str,
    out_of_sample_start: str | pd.Timestamp,
    config: WalkForwardConfig = WalkForwardConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run an out-of-sample, weekly-rebalanced SVR experiment.

    A target dated ``t`` is not available until ``horizon`` trading days later;
    training therefore ends at ``decision_date - horizon``.  The returned
    frames are daily strategy returns, rebalance weights, predictions, and
    summary metrics.
    """
    features = features.copy()
    features["Date"] = pd.to_datetime(features["Date"])
    daily_returns = daily_returns.copy()
    daily_returns.index = pd.to_datetime(daily_returns.index)
    daily_returns = daily_returns.sort_index().dropna(how="all")
    tickers = sorted(t for t in features["ticker"].unique() if t != benchmark)
    dates = _common_feature_dates(features, tickers)
    rebalance_dates = dates[dates >= pd.Timestamp(out_of_sample_start)][::config.rebalance_every]
    if len(rebalance_dates) < 2:
        raise ValueError("The out-of-sample period has fewer than two rebalance dates.")

    models: dict[str, Any] = {}
    last_refit: pd.Timestamp | None = None
    previous_weights = pd.Series(0.0, index=tickers)
    portfolio_parts: list[pd.Series] = []
    equal_parts: list[pd.Series] = []
    benchmark_parts: list[pd.Series] = []
    weights_history: list[pd.Series] = []
    prediction_history: list[dict[str, Any]] = []

    for i, decision_date in enumerate(rebalance_dates[:-1]):
        cutoff = decision_date - pd.offsets.BDay(config.horizon)
        if last_refit is None or (decision_date - last_refit).days >= config.refit_every:
            models = {}
            for ticker in tickers:
                train = features.loc[
                    (features["ticker"] == ticker) & (features["Date"] <= cutoff),
                    feature_columns + [target_column],
                ].dropna()
                if len(train) >= config.min_training_rows:
                    search = make_svr_search(config)
                    search.fit(train[feature_columns], train[target_column])
                    models[ticker] = search
            last_refit = decision_date

        expected: dict[str, float] = {}
        for ticker, model in models.items():
            row = features.loc[(features["ticker"] == ticker) & (features["Date"] == decision_date), feature_columns]
            if row.empty:
                continue
            predicted = float(model.predict(row)[0])
            expected[ticker] = _annualize_horizon_return(predicted, config.horizon)
            prediction_history.append({
                "decision_date": decision_date,
                "ticker": ticker,
                "predicted_horizon_return": predicted,
                "expected_annual_return": expected[ticker],
            })
        if len(expected) != len(tickers):
            continue

        history = daily_returns.loc[:decision_date, tickers].tail(config.covariance_lookback)
        weights = _portfolio_weights(pd.Series(expected), history, config).reindex(tickers).fillna(0.0)
        turnover = float((weights - previous_weights).abs().sum())
        weights_history.append(weights.rename(decision_date))

        next_date = rebalance_dates[i + 1]
        holding = daily_returns.loc[(daily_returns.index > decision_date) & (daily_returns.index <= next_date), tickers]
        if holding.empty:
            continue
        model_returns = holding.mul(weights, axis=1).sum(axis=1)
        model_returns.iloc[0] -= turnover * config.transaction_cost_bps / 10_000.0
        portfolio_parts.append(model_returns.rename("SVR walk-forward"))
        equal_parts.append(holding.mean(axis=1).rename("Equal weight"))
        benchmark_parts.append(daily_returns.loc[holding.index, benchmark].rename("S&P 500"))
        previous_weights = weights

    if not portfolio_parts:
        raise RuntimeError("No holding periods were produced; inspect dates and min_training_rows.")
    backtest = pd.concat([pd.concat(portfolio_parts), pd.concat(equal_parts), pd.concat(benchmark_parts)], axis=1).sort_index()
    weights = pd.DataFrame(weights_history)
    predictions = pd.DataFrame(prediction_history)
    metrics = pd.DataFrame({column: _metrics(backtest[column], config.risk_free_rate) for column in backtest}).T
    return backtest, weights, predictions, metrics


def save_results(output_dir: str | Path, backtest: pd.DataFrame, weights: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Save auditable output files separate from exploratory notebook results."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(output / "daily_returns_walk_forward_svr.csv")
    (1.0 + backtest).cumprod().to_csv(output / "cumulative_returns_walk_forward_svr.csv")
    weights.to_csv(output / "weights_by_rebalance_svr.csv", index_label="decision_date")
    predictions.to_csv(output / "predictions_by_rebalance_svr.csv", index=False)
    metrics.to_csv(output / "summary_walk_forward_svr.csv")
