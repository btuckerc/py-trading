"""Analyze strategy performance by market regime.

This script:
1. Fits regime labels using RegimeLabelGenerator
2. Loads backtest equity curves
3. Computes regime-conditional performance metrics
4. Compares regime-aware vs baseline strategies
5. Generates comprehensive risk reports
"""

import sys
from pathlib import Path
from datetime import date, datetime
import argparse
import pandas as pd
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage import StorageBackend
from data.asof_api import AsOfQueryAPI
from labels.regimes import RegimeLabelGenerator, REGIME_DESCRIPTORS
from backtest.metrics import PerformanceMetrics
from configs.loader import get_config
from loguru import logger


def compute_crisis_period_metrics(
    equity_curve: pd.DataFrame,
    crisis_periods: dict
) -> dict:
    """
    Compute metrics for specific crisis periods.
    
    Args:
        equity_curve: DataFrame with date, equity, returns columns
        crisis_periods: Dict mapping period name -> (start_date, end_date)
    
    Returns:
        Dict mapping period name -> metrics
    """
    results = {}
    equity_curve = equity_curve.copy()
    equity_curve['date'] = pd.to_datetime(equity_curve['date']).dt.date
    
    for period_name, (start, end) in crisis_periods.items():
        if isinstance(start, str):
            start = datetime.strptime(start, "%Y-%m-%d").date()
        if isinstance(end, str):
            end = datetime.strptime(end, "%Y-%m-%d").date()
        
        period_data = equity_curve[
            (equity_curve['date'] >= start) &
            (equity_curve['date'] <= end)
        ]
        
        if len(period_data) > 0:
            metrics = PerformanceMetrics.compute_metrics(period_data)
            metrics['num_days'] = len(period_data)
            results[period_name] = metrics
        else:
            results[period_name] = {'error': 'No data for period'}
    
    return results


def compute_transition_analysis(regimes_df: pd.DataFrame) -> dict:
    """
    Analyze regime transitions.
    
    Returns:
        Dict with transition matrix and average regime duration
    """
    regimes_df = regimes_df.copy()
    regimes_df = regimes_df.sort_values('date')
    
    # Transition matrix
    transitions = {}
    prev_regime = None
    regime_durations = {}
    current_duration = 0
    
    for _, row in regimes_df.iterrows():
        current_regime = row['regime_descriptor']
        
        if prev_regime is not None:
            key = f"{prev_regime} -> {current_regime}"
            transitions[key] = transitions.get(key, 0) + 1
            
            if current_regime != prev_regime:
                # Record duration of previous regime
                if prev_regime not in regime_durations:
                    regime_durations[prev_regime] = []
                regime_durations[prev_regime].append(current_duration)
                current_duration = 1
            else:
                current_duration += 1
        else:
            current_duration = 1
        
        prev_regime = current_regime
    
    # Record final regime duration
    if prev_regime is not None:
        if prev_regime not in regime_durations:
            regime_durations[prev_regime] = []
        regime_durations[prev_regime].append(current_duration)
    
    # Compute average durations
    avg_durations = {
        regime: np.mean(durations) if durations else 0
        for regime, durations in regime_durations.items()
    }
    
    return {
        'transition_counts': transitions,
        'average_duration_days': avg_durations,
    }


def generate_exposure_recommendations(regime_metrics: dict) -> dict:
    """
    Generate recommended exposure multipliers based on regime performance.
    
    Returns:
        Dict mapping regime_descriptor -> recommended_exposure
    """
    recommendations = {}
    
    # Find the best and worst performing regimes
    sharpe_by_regime = {}
    for regime_id, metrics in regime_metrics.items():
        descriptor = metrics.get('regime_descriptor', f'regime_{regime_id}')
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 1.0)
        
        # Risk-adjusted score: higher Sharpe and lower DD is better
        if max_dd > 0:
            risk_score = sharpe / max_dd
        else:
            risk_score = sharpe
        
        sharpe_by_regime[descriptor] = {
            'sharpe': sharpe,
            'max_dd': max_dd,
            'risk_score': risk_score
        }
    
    # Normalize to exposure recommendations
    if sharpe_by_regime:
        max_score = max(s['risk_score'] for s in sharpe_by_regime.values())
        min_score = min(s['risk_score'] for s in sharpe_by_regime.values())
        score_range = max_score - min_score if max_score != min_score else 1.0
        
        for descriptor, scores in sharpe_by_regime.items():
            # Map risk score to 0.25-1.0 exposure range
            normalized = (scores['risk_score'] - min_score) / score_range
            recommended_exposure = 0.25 + 0.75 * normalized
            
            recommendations[descriptor] = {
                'recommended_exposure': round(recommended_exposure, 2),
                'sharpe': round(scores['sharpe'], 3),
                'max_dd': round(scores['max_dd'], 3),
            }
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Analyze strategy performance by market regime")
    parser.add_argument("--start-date", type=str, required=True, help="Start date for regime analysis (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date for regime analysis (YYYY-MM-DD)")
    parser.add_argument("--equity-curve", type=str, help="Path to equity curve CSV (default: look for latest)")
    parser.add_argument("--benchmark-curve", type=str, help="Path to benchmark (SPY) equity curve CSV")
    parser.add_argument("--n-regimes", type=int, default=4, help="Number of regimes to fit")
    parser.add_argument("--regime-method", type=str, default="kmeans", choices=["kmeans", "gmm"], help="Regime clustering method")
    parser.add_argument("--save-regimes", action="store_true", help="Save regime labels to database")
    parser.add_argument("--save-model", type=str, help="Path to save fitted regime model")
    parser.add_argument("--compare-strategies", action="store_true", help="Compare multiple strategy equity curves")
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    logger.info(f"Regime analysis period: {start_date} to {end_date}")
    
    # Initialize storage
    config = get_config()
    storage = StorageBackend(
        db_path=config.database.duckdb_path,
        data_root=config.database.data_root
    )
    
    # Fit regimes
    logger.info("Fitting regime labels...")
    regime_generator = RegimeLabelGenerator(storage, n_regimes=args.n_regimes)
    regimes_df = regime_generator.fit_regimes(start_date, end_date, method=args.regime_method)
    
    if len(regimes_df) == 0:
        logger.error("No regimes generated")
        return
    
    logger.info(f"Fitted {args.n_regimes} regimes over {len(regimes_df)} dates")
    
    # Print regime summary with statistics
    print("\n" + "="*80)
    print("REGIME SUMMARY")
    print("="*80)
    
    regime_counts = regimes_df['regime_descriptor'].value_counts()
    for descriptor in REGIME_DESCRIPTORS:
        count = regime_counts.get(descriptor, 0)
        pct = count / len(regimes_df) * 100 if len(regimes_df) > 0 else 0
        
        # Get regime stats
        stats = regime_generator.regime_stats.get(
            regimes_df[regimes_df['regime_descriptor'] == descriptor]['regime_id'].iloc[0]
            if count > 0 else -1,
            {}
        )
        
        print(f"\n{descriptor}:")
        print(f"  Days: {count} ({pct:.1f}%)")
        if stats:
            print(f"  Median 20d Return: {stats.get('median_return_20d', 0):.2%}")
            print(f"  Median Volatility: {stats.get('median_vol', 0):.2%}")
            print(f"  Mean Drawdown: {stats.get('mean_drawdown', 0):.2%}")
    
    # Save regimes if requested
    if args.save_regimes:
        logger.info("Saving regime labels to database...")
        regime_generator.save_regimes(regimes_df)
    
    # Save model if requested
    if args.save_model:
        regime_generator.save_model(args.save_model)
    
    # Load equity curve
    if args.equity_curve:
        equity_curve_path = Path(args.equity_curve)
    else:
        # Find latest equity curve
        results_dir = Path("artifacts") / "backtest_results"
        equity_curves = list(results_dir.glob("equity_curve_*.csv"))
        if len(equity_curves) == 0:
            logger.error("No equity curve found. Please specify --equity-curve")
            storage.close()
            return
        equity_curve_path = max(equity_curves, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using latest equity curve: {equity_curve_path}")
    
    equity_curve = pd.read_csv(equity_curve_path)
    equity_curve['date'] = pd.to_datetime(equity_curve['date']).dt.date
    
    logger.info(f"Loaded equity curve: {len(equity_curve)} days")
    
    # Merge equity curve with regimes
    merged = equity_curve.merge(regimes_df, on='date', how='inner')
    
    if len(merged) == 0:
        logger.error("No overlap between equity curve and regime dates")
        storage.close()
        return
    
    logger.info(f"Merged data: {len(merged)} days")
    
    # Compute regime-conditional metrics
    logger.info("\nComputing regime-conditional performance metrics...")
    regime_metrics = PerformanceMetrics.compute_regime_metrics(equity_curve, regimes_df)
    
    # Add regime descriptors to metrics
    for regime_id in regime_metrics.keys():
        descriptor = regimes_df[regimes_df['regime_id'] == regime_id]['regime_descriptor'].iloc[0]
        regime_metrics[regime_id]['regime_descriptor'] = descriptor
    
    # Print detailed results
    print("\n" + "="*80)
    print("REGIME-CONDITIONAL PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Strategy: {equity_curve_path.name}")
    print(f"Number of regimes: {args.n_regimes}")
    
    for regime_id in sorted(regime_metrics.keys()):
        metrics = regime_metrics[regime_id]
        descriptor = metrics.get('regime_descriptor', f'regime_{regime_id}')
        regime_days = len(merged[merged['regime_id'] == regime_id])
        
        print(f"\n{'-'*80}")
        print(f"Regime {regime_id}: {descriptor} ({regime_days} days)")
        print(f"{'-'*80}")
        print(f"  Total Return:     {metrics.get('total_return', 0):>10.2%}")
        print(f"  CAGR:             {metrics.get('cagr', 0):>10.2%}")
        print(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):>10.3f}")
        print(f"  Max Drawdown:     {metrics.get('max_drawdown', 0):>10.2%}")
        print(f"  Calmar Ratio:     {metrics.get('calmar_ratio', 0):>10.3f}")
        print(f"  Ann. Volatility:  {metrics.get('annualized_volatility', 0):>10.2%}")
        print(f"  Hit Rate:         {metrics.get('hit_rate', 0):>10.2%}")
        print(f"  VaR (5%):         {metrics.get('var_5pct', 0):>10.2%}")
        print(f"  CVaR (5%):        {metrics.get('cvar_5pct', 0):>10.2%}")
    
    # Overall metrics for comparison
    overall_metrics = PerformanceMetrics.compute_metrics(equity_curve)
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*80}")
    print(f"  Total Return:     {overall_metrics['total_return']:>10.2%}")
    print(f"  CAGR:             {overall_metrics['cagr']:>10.2%}")
    print(f"  Sharpe Ratio:     {overall_metrics['sharpe_ratio']:>10.3f}")
    print(f"  Max Drawdown:     {overall_metrics['max_drawdown']:>10.2%}")
    print(f"  Calmar Ratio:     {overall_metrics['calmar_ratio']:>10.3f}")
    print(f"  Ann. Volatility:  {overall_metrics['annualized_volatility']:>10.2%}")
    
    # Crisis period analysis
    crisis_periods = {
        'COVID_crash_2020': ('2020-02-19', '2020-03-23'),
        'COVID_recovery_2020': ('2020-03-24', '2020-08-31'),
        'Bear_market_2022': ('2022-01-03', '2022-10-12'),
        'Recovery_2023': ('2023-01-01', '2023-12-31'),
    }
    
    print(f"\n{'='*80}")
    print("CRISIS PERIOD ANALYSIS")
    print(f"{'='*80}")
    
    crisis_metrics = compute_crisis_period_metrics(equity_curve, crisis_periods)
    for period_name, metrics in crisis_metrics.items():
        if 'error' not in metrics:
            print(f"\n{period_name}:")
            print(f"  Return: {metrics.get('total_return', 0):.2%}")
            print(f"  Max DD: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.3f}")
    
    # Regime transition analysis
    print(f"\n{'='*80}")
    print("REGIME TRANSITION ANALYSIS")
    print(f"{'='*80}")
    
    transition_analysis = compute_transition_analysis(regimes_df)
    
    print("\nAverage Regime Duration (days):")
    for regime, duration in transition_analysis['average_duration_days'].items():
        print(f"  {regime}: {duration:.1f} days")
    
    # Exposure recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDED EXPOSURE MULTIPLIERS")
    print(f"{'='*80}")
    print("(Based on historical risk-adjusted performance)")
    
    recommendations = generate_exposure_recommendations(regime_metrics)
    for descriptor, rec in recommendations.items():
        print(f"\n{descriptor}:")
        print(f"  Recommended Exposure: {rec['recommended_exposure']:.0%}")
        print(f"  Historical Sharpe: {rec['sharpe']:.3f}")
        print(f"  Historical Max DD: {rec['max_dd']:.2%}")
    
    # Save results
    output_path = Path("artifacts") / "regime_analysis"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save regime labels
    regimes_df.to_csv(output_path / f"regimes_{start_date}_{end_date}.csv", index=False)
    
    # Save comprehensive regime metrics
    regime_summary = {
        'period': {
            'start_date': str(start_date),
            'end_date': str(end_date)
        },
        'regime_config': {
            'n_regimes': args.n_regimes,
            'method': args.regime_method
        },
        'overall_metrics': overall_metrics,
        'regime_metrics': {str(k): v for k, v in regime_metrics.items()},
        'crisis_period_metrics': crisis_metrics,
        'transition_analysis': transition_analysis,
        'exposure_recommendations': recommendations,
        'regime_stats': {str(k): v for k, v in regime_generator.regime_stats.items()},
    }
    
    summary_path = output_path / f"regime_metrics_{start_date}_{end_date}.json"
    with open(summary_path, 'w') as f:
        json.dump(regime_summary, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"  Regime labels: regimes_{start_date}_{end_date}.csv")
    logger.info(f"  Regime metrics: regime_metrics_{start_date}_{end_date}.json")
    
    # Actionable recommendations
    print(f"\n{'='*80}")
    print("ACTIONABLE RECOMMENDATIONS")
    print(f"{'='*80}")
    print("""
1. REGIME-AWARE EXPOSURE:
   - Use the recommended exposure multipliers in portfolio.regime_policy
   - Current config supports automatic exposure scaling by regime

2. DEFENSIVE SECTOR ROTATION:
   - In bear_high_vol: Overweight Consumer Staples, Health Care, Utilities
   - In bull_low_vol: Allow higher Tech and Consumer Discretionary exposure
   - Current config supports automatic sector tilts by regime

3. VIX-STYLE VOLATILITY SCALING:
   - Target ~15% portfolio volatility
   - Scale down positions when realized vol exceeds target
   - Current config supports volatility_scaling in portfolio section

4. DRAWDOWN THROTTLING:
   - Start reducing exposure at 15% drawdown
   - Minimum 25% exposure at 25% drawdown
   - Current config supports drawdown throttle

5. MONITORING:
   - Watch for regime transitions (avg duration shown above)
   - Review performance weekly by current regime
   - Consider disabling shorts in bear_high_vol if using long/short
""")
    print("="*80)
    
    storage.close()


if __name__ == "__main__":
    main()

