#!/usr/bin/env python3
"""
Generate professional performance reports with charts and tables.

This script takes simulation results (from simulate_daily_trading.py) and
generates industry-standard performance visualizations and summary tables.

Usage:
    # Generate report from a date range (runs simulation internally)
    python scripts/generate_performance_report.py --start-date 2025-09-01 --end-date 2025-11-26

    # Generate from existing JSON results
    python scripts/generate_performance_report.py --from-json results.json

    # Specify output directory
    python scripts/generate_performance_report.py --start-date 2025-09-01 --end-date 2025-11-26 --output-dir reports/

    # Generate HTML report
    python scripts/generate_performance_report.py --start-date 2025-09-01 --end-date 2025-11-26 --html
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate performance reports with charts and tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--from-json", type=str,
        help="Load results from JSON file (output of simulate_daily_trading.py --output-json)"
    )
    input_group.add_argument(
        "--start-date", type=str,
        help="Start date for simulation (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date", type=str,
        help="End date for simulation (YYYY-MM-DD), required if --start-date is used"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top stocks to hold (default: 5)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", type=str, default="artifacts/reports",
        help="Output directory for reports (default: artifacts/reports)"
    )
    parser.add_argument(
        "--report-name", type=str, default=None,
        help="Base name for report files (default: auto-generated from dates)"
    )
    parser.add_argument(
        "--html", action="store_true",
        help="Generate HTML report in addition to images"
    )
    parser.add_argument(
        "--no-charts", action="store_true",
        help="Skip chart generation (tables only)"
    )
    parser.add_argument(
        "--dark-mode", action="store_true",
        help="Use dark theme for charts"
    )
    parser.add_argument(
        "--benchmark", type=str,
        help="Single benchmark to show (e.g., 'sp500', 'dow', 'nasdaq', or ticker like 'SPY')"
    )
    parser.add_argument(
        "--benchmarks", type=str,
        help="Comma-separated list of benchmarks to show (e.g., 'sp500,dow,nasdaq' or 'SPY,QQQ')"
    )
    
    return parser.parse_args()


def run_simulation(start_date: str, end_date: str, top_k: int = 5) -> dict:
    """Run the simulation and return results."""
    from scripts.simulate_daily_trading import DailyTradingSimulator
    from configs.loader import get_config
    from data.storage import StorageBackend
    
    config = get_config()
    storage = StorageBackend(
        db_path=config.database.duckdb_path,
        data_root=config.database.data_root
    )
    
    try:
        simulator = DailyTradingSimulator(
            storage=storage,
            config=config.__dict__,
            top_k=top_k,
            train_days=252,
            initial_capital=100000
        )
        
        results = simulator.simulate(
            start_date=datetime.strptime(start_date, "%Y-%m-%d").date(),
            end_date=datetime.strptime(end_date, "%Y-%m-%d").date(),
            retrain_frequency=0,
            verbose=False
        )
        return results
    finally:
        storage.close()


def calculate_metrics(results: dict) -> dict:
    """Calculate additional performance metrics."""
    daily = results["daily_results"]
    summary = results["summary"]
    
    # Extract returns
    portfolio_returns = [d["portfolio_return"] for d in daily]
    
    # Extract benchmark returns (support both old spy_return and new benchmark_returns)
    benchmark_returns_dict = {}
    spy_returns = []  # Backward compatibility
    
    # Get benchmark names from config
    from configs.loader import get_config
    config = get_config()
    benchmark_config = getattr(config, 'benchmarks', {})
    benchmark_definitions = benchmark_config.get('definitions', {})
    default_benchmarks = benchmark_config.get('default', ['sp500'])
    
    # Map benchmark names to tickers and display names
    benchmark_tickers = []
    benchmark_display_names = {}
    for bench_name in default_benchmarks:
        if bench_name in benchmark_definitions:
            ticker = benchmark_definitions[bench_name]['ticker']
            display_name = benchmark_definitions[bench_name]['name']
            benchmark_tickers.append(ticker)
            benchmark_display_names[ticker] = display_name
    
    # Fallback to SPY if no benchmarks configured
    if not benchmark_tickers:
        benchmark_tickers = ['SPY']
        benchmark_display_names['SPY'] = 'S&P 500'
    
    # Extract benchmark returns from daily results
    for ticker in benchmark_tickers:
        returns = []
        for d in daily:
            # Try new format first
            if "benchmark_returns" in d and ticker in d["benchmark_returns"]:
                returns.append(d["benchmark_returns"][ticker])
            # Fallback to old spy_return for SPY
            elif ticker == "SPY" and "spy_return" in d and d["spy_return"] is not None:
                returns.append(d["spy_return"])
            else:
                returns.append(None)
        
        # Filter out None values and align with portfolio returns
        valid_returns = [r for r in returns if r is not None]
        if valid_returns:
            benchmark_returns_dict[ticker] = {
                "returns": returns,
                "valid_returns": valid_returns,
                "display_name": benchmark_display_names.get(ticker, ticker)
            }
            if ticker == "SPY":
                spy_returns = valid_returns
    
    # Calculate cumulative returns
    portfolio_cumulative = np.cumprod([1 + r for r in portfolio_returns])
    
    benchmark_cumulative = {}
    for ticker, bench_data in benchmark_returns_dict.items():
        valid_ret = bench_data["valid_returns"]
        cumulative = np.cumprod([1 + r for r in valid_ret])
        benchmark_cumulative[ticker] = cumulative.tolist()
    
    # Backward compatibility: spy_cumulative
    spy_cumulative = benchmark_cumulative.get("SPY", [])
    
    # Calculate rolling metrics (21-day = ~1 month)
    window = min(21, len(portfolio_returns) // 2) if len(portfolio_returns) > 10 else len(portfolio_returns)
    
    # Rolling Sharpe (annualized)
    if len(portfolio_returns) >= window:
        rolling_returns = pd.Series(portfolio_returns)
        rolling_mean = rolling_returns.rolling(window).mean() * 252
        rolling_std = rolling_returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mean / rolling_std).fillna(0).tolist()
    else:
        rolling_sharpe = [0] * len(portfolio_returns)
    
    # Drawdown series
    running_max = np.maximum.accumulate(portfolio_cumulative)
    drawdown = (portfolio_cumulative - running_max) / running_max
    
    # Monthly returns (if enough data)
    dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in daily]
    
    # Build DataFrame with portfolio and all benchmark returns
    df_data = {"date": dates, "return": portfolio_returns}
    for ticker, bench_data in benchmark_returns_dict.items():
        df_data[f"{ticker}_return"] = bench_data["returns"][:len(portfolio_returns)]
    
    # Backward compatibility: spy_return column
    if "SPY" in benchmark_returns_dict:
        df_data["spy_return"] = benchmark_returns_dict["SPY"]["returns"][:len(portfolio_returns)]
    else:
        df_data["spy_return"] = [None] * len(portfolio_returns)
    
    df = pd.DataFrame(df_data)
    df["month"] = df["date"].dt.to_period("M")
    
    # Aggregate monthly returns
    monthly_agg = {"return": lambda x: np.prod(1 + x) - 1}
    for ticker in benchmark_returns_dict.keys():
        monthly_agg[f"{ticker}_return"] = lambda x: np.prod(1 + x.fillna(0)) - 1
    monthly_agg["spy_return"] = lambda x: np.prod(1 + x.fillna(0)) - 1  # Backward compatibility
    
    monthly = df.groupby("month").agg(monthly_agg).reset_index()
    monthly["month_str"] = monthly["month"].astype(str)
    
    # Best/worst days
    sorted_returns = sorted(enumerate(portfolio_returns), key=lambda x: x[1])
    worst_days = [(daily[i]["date"], r) for i, r in sorted_returns[:5]]
    best_days = [(daily[i]["date"], r) for i, r in sorted_returns[-5:][::-1]]
    
    # Position frequency
    position_counts = {}
    for d in daily:
        for pos in d["positions"]:
            symbol = pos["symbol"]
            position_counts[symbol] = position_counts.get(symbol, 0) + 1
    top_positions = sorted(position_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "portfolio_cumulative": portfolio_cumulative.tolist(),
        "spy_cumulative": spy_cumulative,  # Backward compatibility
        "benchmark_cumulative": benchmark_cumulative,  # New: all benchmarks
        "benchmark_display_names": benchmark_display_names,  # For labels
        "drawdown": drawdown.tolist(),
        "rolling_sharpe": rolling_sharpe,
        "monthly_returns": monthly.to_dict("records"),
        "best_days": best_days,
        "worst_days": worst_days,
        "top_positions": top_positions,
        "dates": [d["date"] for d in daily],
    }


def setup_style(dark_mode: bool = False):
    """Set up matplotlib style."""
    if dark_mode:
        plt.style.use('dark_background')
        colors = {
            "portfolio": "#00D4AA",  # Teal
            "spy": "#FF6B6B",        # Coral
            "positive": "#00D4AA",
            "negative": "#FF6B6B",
            "neutral": "#888888",
            "grid": "#333333",
            "text": "#FFFFFF",
            "bg": "#1a1a2e",
        }
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = {
            "portfolio": "#2E86AB",  # Blue
            "spy": "#A23B72",        # Magenta
            "positive": "#28A745",
            "negative": "#DC3545",
            "neutral": "#6C757D",
            "grid": "#E0E0E0",
            "text": "#212529",
            "bg": "#FFFFFF",
        }
    return colors


def generate_equity_curve(results: dict, metrics: dict, colors: dict, output_path: Path):
    """Generate equity curve chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in metrics["dates"]]
    
    # Plot equity curves
    portfolio_equity = [100000 * c for c in metrics["portfolio_cumulative"]]
    ax.plot(dates, portfolio_equity, label="ML Strategy", color=colors["portfolio"], linewidth=2)
    
    # Plot all benchmarks
    benchmark_colors = {
        "SPY": colors["spy"],
        "DIA": "#FFA500",  # Orange
        "QQQ": "#00CED1",  # Dark turquoise
    }
    linestyles = ["--", "-.", ":"]
    
    benchmark_cumulative = metrics.get("benchmark_cumulative", {})
    benchmark_display_names = metrics.get("benchmark_display_names", {})
    
    # Fallback to spy_cumulative for backward compatibility
    if not benchmark_cumulative and "spy_cumulative" in metrics:
        spy_cumulative = metrics["spy_cumulative"]
        if spy_cumulative:
            benchmark_cumulative["SPY"] = spy_cumulative
            benchmark_display_names["SPY"] = "S&P 500"
    
    for i, (ticker, cumulative) in enumerate(benchmark_cumulative.items()):
        if cumulative:
            equity = [100000 * c for c in cumulative[:len(dates)]]
            display_name = benchmark_display_names.get(ticker, ticker)
            color = benchmark_colors.get(ticker, colors["spy"])
            linestyle = linestyles[i % len(linestyles)]
            ax.plot(dates[:len(equity)], equity, 
                   label=f"{display_name} ({ticker})", 
                   color=color, linewidth=2, linestyle=linestyle)
    
    # Fill between
    ax.fill_between(dates, portfolio_equity, alpha=0.1, color=colors["portfolio"])
    
    # Formatting
    ax.set_title("Portfolio Equity Curve", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Add final values annotation
    final_portfolio = portfolio_equity[-1]
    ax.annotate(f"${final_portfolio:,.0f}", xy=(dates[-1], final_portfolio),
                xytext=(10, 0), textcoords="offset points", fontsize=10, color=colors["portfolio"])
    
    # Annotate final benchmark values
    y_offset = 0
    for ticker, cumulative in benchmark_cumulative.items():
        if cumulative:
            equity = [100000 * c for c in cumulative[:len(dates)]]
            if equity:
                final_value = equity[-1]
                color = benchmark_colors.get(ticker, colors["spy"])
                ax.annotate(f"${final_value:,.0f}", xy=(dates[-1], final_value),
                           xytext=(10, y_offset), textcoords="offset points", 
                           fontsize=10, color=color)
                y_offset -= 15
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def generate_drawdown_chart(results: dict, metrics: dict, colors: dict, output_path: Path):
    """Generate drawdown chart."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in metrics["dates"]]
    drawdown = [d * 100 for d in metrics["drawdown"]]  # Convert to percentage
    
    ax.fill_between(dates, drawdown, 0, alpha=0.7, color=colors["negative"])
    ax.plot(dates, drawdown, color=colors["negative"], linewidth=1)
    
    # Mark max drawdown
    min_dd_idx = np.argmin(drawdown)
    ax.scatter([dates[min_dd_idx]], [drawdown[min_dd_idx]], color=colors["negative"], s=100, zorder=5)
    ax.annotate(f"Max: {drawdown[min_dd_idx]:.1f}%", 
                xy=(dates[min_dd_idx], drawdown[min_dd_idx]),
                xytext=(10, -20), textcoords="offset points", fontsize=10)
    
    ax.set_title("Portfolio Drawdown", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min(drawdown) * 1.2, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def generate_monthly_returns_heatmap(results: dict, metrics: dict, colors: dict, output_path: Path):
    """Generate monthly returns comparison bar chart."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    monthly = metrics["monthly_returns"]
    if len(monthly) < 2:
        # Not enough data for monthly chart
        ax.text(0.5, 0.5, "Not enough data for monthly returns", 
                ha="center", va="center", transform=ax.transAxes)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    
    months = [m["month_str"] for m in monthly]
    portfolio_returns = [m["return"] * 100 for m in monthly]
    
    # Get benchmark returns
    benchmark_colors = {
        "SPY": colors["spy"],
        "DIA": "#FFA500",  # Orange
        "QQQ": "#00CED1",  # Dark turquoise
    }
    benchmark_display_names = metrics.get("benchmark_display_names", {})
    
    # Find which benchmarks are available in monthly data
    available_benchmarks = []
    for ticker in ["SPY", "DIA", "QQQ"]:
        col_name = f"{ticker}_return"
        if col_name in monthly[0]:
            available_benchmarks.append(ticker)
    
    # Fallback to spy_return for backward compatibility
    if not available_benchmarks and "spy_return" in monthly[0]:
        available_benchmarks = ["SPY"]
    
    x = np.arange(len(months))
    n_bars = 1 + len(available_benchmarks)  # Portfolio + benchmarks
    width = 0.8 / n_bars
    
    # Plot portfolio
    offset = -(n_bars - 1) * width / 2
    bars_portfolio = ax.bar(x + offset, portfolio_returns, width, 
                           label="ML Strategy", color=colors["portfolio"])
    offset += width
    
    # Plot benchmarks
    benchmark_bars = {}
    for ticker in available_benchmarks:
        col_name = f"{ticker}_return" if ticker != "SPY" or f"{ticker}_return" in monthly[0] else "spy_return"
        bench_returns = [m.get(col_name, 0) * 100 for m in monthly]
        display_name = benchmark_display_names.get(ticker, ticker)
        color = benchmark_colors.get(ticker, colors["spy"])
        bars = ax.bar(x + offset, bench_returns, width, 
                     label=display_name, color=color)
        benchmark_bars[ticker] = bars
        offset += width
    
    # Color bars by positive/negative
    for bar, val in zip(bars_portfolio, portfolio_returns):
        if val < 0:
            bar.set_alpha(0.7)
    for ticker, bars in benchmark_bars.items():
        col_name = f"{ticker}_return" if ticker != "SPY" or f"{ticker}_return" in monthly[0] else "spy_return"
        bench_returns = [m.get(col_name, 0) * 100 for m in monthly]
        for bar, val in zip(bars, bench_returns):
            if val < 0:
                bar.set_alpha(0.7)
    
    ax.set_title("Monthly Returns Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Return (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.axhline(y=0, color=colors["neutral"], linestyle="-", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def generate_returns_distribution(results: dict, metrics: dict, colors: dict, output_path: Path):
    """Generate returns distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    daily = results["daily_results"]
    returns = [d["portfolio_return"] * 100 for d in daily]
    
    # Histogram
    n, bins, patches = ax.hist(returns, bins=30, edgecolor="white", alpha=0.7)
    
    # Color by positive/negative
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor(colors["negative"])
        else:
            patch.set_facecolor(colors["positive"])
    
    # Add vertical line at mean
    mean_return = np.mean(returns)
    ax.axvline(x=mean_return, color=colors["portfolio"], linestyle="--", linewidth=2, label=f"Mean: {mean_return:.2f}%")
    ax.axvline(x=0, color=colors["neutral"], linestyle="-", linewidth=1)
    
    ax.set_title("Daily Returns Distribution", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Daily Return (%)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def generate_position_frequency(results: dict, metrics: dict, colors: dict, output_path: Path):
    """Generate top positions bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_positions = metrics["top_positions"][:15]  # Top 15
    if not top_positions:
        ax.text(0.5, 0.5, "No position data", ha="center", va="center", transform=ax.transAxes)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    
    symbols = [p[0] for p in top_positions]
    counts = [p[1] for p in top_positions]
    
    bars = ax.barh(symbols[::-1], counts[::-1], color=colors["portfolio"], alpha=0.8)
    
    ax.set_title("Most Frequently Held Positions", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Days Held", fontsize=11)
    ax.set_ylabel("Symbol", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add count labels
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                str(count), va="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def generate_summary_dashboard(results: dict, metrics: dict, colors: dict, output_path: Path):
    """Generate a single-page summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    summary = results["summary"]
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in metrics["dates"]]
    
    # 1. Equity Curve (top, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    portfolio_equity = [100000 * c for c in metrics["portfolio_cumulative"]]
    ax1.plot(dates, portfolio_equity, label="ML Strategy", color=colors["portfolio"], linewidth=2)
    
    # Plot all benchmarks
    benchmark_colors = {
        "SPY": colors["spy"],
        "DIA": "#FFA500",
        "QQQ": "#00CED1",
    }
    linestyles = ["--", "-.", ":"]
    benchmark_cumulative = metrics.get("benchmark_cumulative", {})
    benchmark_display_names = metrics.get("benchmark_display_names", {})
    
    # Fallback to spy_cumulative for backward compatibility
    if not benchmark_cumulative and "spy_cumulative" in metrics:
        spy_cumulative = metrics["spy_cumulative"]
        if spy_cumulative:
            benchmark_cumulative["SPY"] = spy_cumulative
            benchmark_display_names["SPY"] = "S&P 500"
    
    for i, (ticker, cumulative) in enumerate(benchmark_cumulative.items()):
        if cumulative:
            equity = [100000 * c for c in cumulative[:len(dates)]]
            display_name = benchmark_display_names.get(ticker, ticker)
            color = benchmark_colors.get(ticker, colors["spy"])
            linestyle = linestyles[i % len(linestyles)]
            ax1.plot(dates[:len(equity)], equity, 
                    label=display_name, color=color, linewidth=2, linestyle=linestyle)
    
    ax1.fill_between(dates, portfolio_equity, alpha=0.1, color=colors["portfolio"])
    ax1.set_title("Equity Curve", fontsize=12, fontweight="bold")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x/1000:.0f}k"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Key Metrics Box (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    
    # Format values with None checks
    spy_return_str = f"{summary['spy_return_pct']:+.2f}%" if summary.get('spy_return_pct') is not None else "N/A"
    alpha_str = f"{summary['alpha_pct']:+.2f}%" if summary.get('alpha_pct') is not None else "N/A"
    
    metrics_text = f"""
    PERFORMANCE SUMMARY
    ═══════════════════════════
    
    Period: {summary['start_date']} to {summary['end_date']}
    Trading Days: {summary['trading_days']}
    
    ML Strategy:     {summary['total_return_pct']:+.2f}%
    S&P 500 (SPY):   {spy_return_str}
    Alpha:           {alpha_str}
    
    Sharpe Ratio:    {summary['sharpe_ratio']:.2f}
    Max Drawdown:    {summary['max_drawdown_pct']:.2f}%
    Win Rate:        {summary['win_rate_pct']:.1f}%
    Volatility:      {summary['annualized_volatility_pct']:.1f}%
    """
    
    ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    
    # 3. Drawdown (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    drawdown = [d * 100 for d in metrics["drawdown"]]
    ax3.fill_between(dates, drawdown, 0, alpha=0.7, color=colors["negative"])
    ax3.set_title("Drawdown", fontsize=12, fontweight="bold")
    ax3.set_ylabel("%")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax3.grid(True, alpha=0.3)
    
    # 4. Returns Distribution (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    daily_returns = [d["portfolio_return"] * 100 for d in results["daily_results"]]
    n, bins, patches = ax4.hist(daily_returns, bins=20, edgecolor="white", alpha=0.7)
    for patch, left_edge in zip(patches, bins[:-1]):
        patch.set_facecolor(colors["positive"] if left_edge >= 0 else colors["negative"])
    ax4.axvline(x=0, color=colors["neutral"], linestyle="-", linewidth=1)
    ax4.set_title("Daily Returns Distribution", fontsize=12, fontweight="bold")
    ax4.set_xlabel("%")
    ax4.grid(True, alpha=0.3, axis="y")
    
    # 5. Top Positions (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    top_pos = metrics["top_positions"][:8]
    if top_pos:
        symbols = [p[0] for p in top_pos]
        counts = [p[1] for p in top_pos]
        ax5.barh(symbols[::-1], counts[::-1], color=colors["portfolio"], alpha=0.8)
        ax5.set_title("Top Positions", fontsize=12, fontweight="bold")
        ax5.set_xlabel("Days Held")
    ax5.grid(True, alpha=0.3, axis="x")
    
    # 6. Monthly Returns (bottom, spans all columns)
    ax6 = fig.add_subplot(gs[2, :])
    monthly = metrics["monthly_returns"]
    if len(monthly) >= 2:
        months = [m["month_str"] for m in monthly]
        portfolio_monthly = [m["return"] * 100 for m in monthly]
        
        # Get available benchmarks
        available_benchmarks = []
        for ticker in ["SPY", "DIA", "QQQ"]:
            col_name = f"{ticker}_return"
            if col_name in monthly[0]:
                available_benchmarks.append(ticker)
        
        # Fallback to spy_return
        if not available_benchmarks and "spy_return" in monthly[0]:
            available_benchmarks = ["SPY"]
        
        x = np.arange(len(months))
        n_bars = 1 + len(available_benchmarks)
        width = 0.8 / n_bars
        
        offset = -(n_bars - 1) * width / 2
        ax6.bar(x + offset, portfolio_monthly, width, label="ML Strategy", color=colors["portfolio"])
        offset += width
        
        benchmark_colors = {
            "SPY": colors["spy"],
            "DIA": "#FFA500",
            "QQQ": "#00CED1",
        }
        benchmark_display_names = metrics.get("benchmark_display_names", {})
        
        for ticker in available_benchmarks:
            col_name = f"{ticker}_return" if ticker != "SPY" or f"{ticker}_return" in monthly[0] else "spy_return"
            bench_monthly = [m.get(col_name, 0) * 100 for m in monthly]
            display_name = benchmark_display_names.get(ticker, ticker)
            color = benchmark_colors.get(ticker, colors["spy"])
            ax6.bar(x + offset, bench_monthly, width, label=display_name, color=color)
            offset += width
        
        ax6.set_xticks(x)
        ax6.set_xticklabels(months, rotation=45, ha="right")
        ax6.axhline(y=0, color=colors["neutral"], linestyle="-", linewidth=0.5)
        ax6.legend(loc="upper left", fontsize=9)
    ax6.set_title("Monthly Returns", fontsize=12, fontweight="bold")
    ax6.set_ylabel("%")
    ax6.grid(True, alpha=0.3, axis="y")
    
    # Main title
    fig.suptitle(f"Trading Strategy Performance Report\n{summary['start_date']} to {summary['end_date']}", 
                 fontsize=16, fontweight="bold", y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def generate_text_summary(results: dict, metrics: dict) -> str:
    """Generate text summary tables."""
    summary = results["summary"]
    
    lines = []
    lines.append("=" * 80)
    lines.append("PERFORMANCE REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Period info
    lines.append(f"Period:        {summary['start_date']} to {summary['end_date']}")
    lines.append(f"Trading Days:  {summary['trading_days']}")
    lines.append(f"Universe:      {summary['universe_size']} assets")
    lines.append(f"Strategy:      Long top-{summary['top_k']} (equal weight, daily rebalance)")
    lines.append("")
    
    # Performance table
    lines.append("-" * 80)
    lines.append("PERFORMANCE METRICS")
    lines.append("-" * 80)
    
    # Get benchmark returns from summary
    benchmark_returns_pct = summary.get('benchmark_returns_pct', {})
    benchmark_display_names = metrics.get("benchmark_display_names", {})
    
    # Backward compatibility: use spy_return_pct if benchmark_returns_pct not available
    # Only create fallback if spy_return_pct is not None
    if not benchmark_returns_pct and 'spy_return_pct' in summary and summary['spy_return_pct'] is not None:
        benchmark_returns_pct = {'SPY': summary['spy_return_pct']}
        benchmark_display_names = {'SPY': 'S&P 500'}
    
    # Filter out None values from benchmark_returns_pct
    benchmark_returns_pct = {k: v for k, v in benchmark_returns_pct.items() if v is not None}
    
    # Build header with benchmark columns (only include benchmarks with valid data)
    header_cols = ['Metric', 'ML Strategy']
    for ticker in ['SPY', 'DIA', 'QQQ']:
        if ticker in benchmark_returns_pct:
            display_name = benchmark_display_names.get(ticker, ticker)
            header_cols.append(display_name)
    
    # Print header
    header_line = f"{header_cols[0]:<30}"
    for col in header_cols[1:]:
        header_line += f" {col:>15}"
    lines.append(header_line)
    lines.append("-" * 80)
    
    # Total Return row
    return_line = f"{'Total Return':<30} {summary['total_return_pct']:>+14.2f}%"
    for ticker in ['SPY', 'DIA', 'QQQ']:
        if ticker in benchmark_returns_pct:
            bench_return = benchmark_returns_pct[ticker]
            if bench_return is not None:
                return_line += f" {bench_return:>+14.2f}%"
            else:
                return_line += f" {'N/A':>15}"
    lines.append(return_line)
    
    # Alpha row (vs primary benchmark)
    if summary.get('alpha_pct') is not None:
        alpha_line = f"{'Alpha (vs Primary)':<30} {summary['alpha_pct']:>+14.2f}%"
        for ticker in ['SPY', 'DIA', 'QQQ']:
            if ticker in benchmark_returns_pct:
                bench_return = benchmark_returns_pct[ticker]
                if bench_return is not None:
                    alpha_vs_bench = (summary['total_return_pct'] - bench_return)
                    alpha_line += f" {alpha_vs_bench:>+14.2f}%"
                else:
                    alpha_line += f" {'N/A':>15}"
        lines.append(alpha_line)
    
    lines.append(f"{'Annualized Volatility':<30} {summary['annualized_volatility_pct']:>14.2f}%")
    lines.append(f"{'Sharpe Ratio':<30} {summary['sharpe_ratio']:>15.2f}")
    lines.append(f"{'Max Drawdown':<30} {summary['max_drawdown_pct']:>14.2f}%")
    lines.append(f"{'Win Rate':<30} {summary['win_rate_pct']:>14.1f}%")
    lines.append("")
    
    # Best/Worst days
    lines.append("-" * 80)
    lines.append("BEST & WORST DAYS")
    lines.append("-" * 80)
    lines.append(f"{'Best Days':<40} {'Worst Days':<40}")
    lines.append("-" * 80)
    
    best = metrics["best_days"]
    worst = metrics["worst_days"]
    for i in range(max(len(best), len(worst))):
        best_str = f"{best[i][0]}: {best[i][1]*100:+.2f}%" if i < len(best) else ""
        worst_str = f"{worst[i][0]}: {worst[i][1]*100:+.2f}%" if i < len(worst) else ""
        lines.append(f"{best_str:<40} {worst_str:<40}")
    lines.append("")
    
    # Top positions
    lines.append("-" * 80)
    lines.append("MOST FREQUENTLY HELD POSITIONS")
    lines.append("-" * 80)
    lines.append(f"{'Symbol':<10} {'Days Held':>10} {'% of Period':>15}")
    lines.append("-" * 40)
    
    total_days = summary['trading_days'] - 1  # -1 because last day has no return
    for symbol, count in metrics["top_positions"][:10]:
        pct = count / total_days * 100
        lines.append(f"{symbol:<10} {count:>10} {pct:>14.1f}%")
    lines.append("")
    
    # Monthly returns
    if len(metrics["monthly_returns"]) >= 2:
        lines.append("-" * 80)
        lines.append("MONTHLY RETURNS")
        lines.append("-" * 80)
        
        # Build header
        monthly_header = f"{'Month':<15} {'ML Strategy':>15}"
        available_benchmarks = []
        for ticker in ['SPY', 'DIA', 'QQQ']:
            col_name = f"{ticker}_return"
            if col_name in metrics["monthly_returns"][0]:
                available_benchmarks.append(ticker)
                display_name = benchmark_display_names.get(ticker, ticker)
                monthly_header += f" {display_name:>15}"
        
        # Fallback to spy_return
        if not available_benchmarks and "spy_return" in metrics["monthly_returns"][0]:
            available_benchmarks = ["SPY"]
            monthly_header += f" {'S&P 500':>15}"
        
        lines.append(monthly_header)
        lines.append("-" * (15 + 15 * (1 + len(available_benchmarks))))
        
        for m in metrics["monthly_returns"]:
            month_line = f"{m['month_str']:<15} {m['return']*100:>+14.2f}%"
            for ticker in available_benchmarks:
                col_name = f"{ticker}_return" if ticker != "SPY" or f"{ticker}_return" in m else "spy_return"
                bench_return = m.get(col_name, 0) * 100
                month_line += f" {bench_return:>+14.2f}%"
            lines.append(month_line)
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def generate_html_report(results: dict, metrics: dict, charts_dir: Path, output_path: Path):
    """Generate HTML report with embedded charts."""
    summary = results["summary"]
    
    # Determine assessment (handle None alpha_pct)
    alpha_pct = summary.get('alpha_pct')
    if alpha_pct is None:
        assessment = ("➖ NO BENCHMARK DATA", "color: #6C757D;")
        alpha_text = ""
    elif alpha_pct > 1:
        assessment = ("✅ OUTPERFORMED", "color: #28A745;")
        alpha_text = f" S&P 500 by {abs(alpha_pct):.2f}%"
    elif alpha_pct < -1:
        assessment = ("❌ UNDERPERFORMED", "color: #DC3545;")
        alpha_text = f" S&P 500 by {abs(alpha_pct):.2f}%"
    else:
        assessment = ("➖ MATCHED", "color: #6C757D;")
        alpha_text = f" S&P 500 by {abs(alpha_pct):.2f}%"
    
    # Format benchmark returns with None checks
    spy_return_pct = summary.get('spy_return_pct')
    spy_return_str = f"{spy_return_pct:+.2f}%" if spy_return_pct is not None else "N/A"
    spy_return_class = "positive" if (spy_return_pct or 0) > 0 else "negative" if spy_return_pct is not None else ""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Report | {summary['start_date']} to {summary['end_date']}</title>
    <style>
        :root {{
            --primary: #2E86AB;
            --secondary: #A23B72;
            --positive: #28A745;
            --negative: #DC3545;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #212529;
            --border: #dee2e6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
        h2 {{ font-size: 1.5rem; margin: 2rem 0 1rem; border-bottom: 2px solid var(--primary); padding-bottom: 0.5rem; }}
        .subtitle {{ color: #6c757d; margin-bottom: 2rem; }}
        .card {{
            background: var(--card-bg);
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        .metric {{
            text-align: center;
            padding: 1rem;
            background: var(--bg);
            border-radius: 8px;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary);
        }}
        .metric-label {{ font-size: 0.9rem; color: #6c757d; }}
        .positive {{ color: var(--positive) !important; }}
        .negative {{ color: var(--negative) !important; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: var(--bg); font-weight: 600; }}
        tr:hover {{ background: #f1f3f4; }}
        .chart-container {{ margin: 1rem 0; text-align: center; }}
        .chart-container img {{ max-width: 100%; height: auto; border-radius: 8px; }}
        .assessment {{ font-size: 1.2rem; font-weight: bold; padding: 1rem; text-align: center; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
        @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
        .footer {{ text-align: center; color: #6c757d; margin-top: 3rem; font-size: 0.9rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Trading Strategy Performance Report</h1>
        <p class="subtitle">{summary['start_date']} to {summary['end_date']} | {summary['trading_days']} trading days | {summary['universe_size']} assets</p>
        
        <div class="card">
            <div class="assessment" style="{assessment[1]}">{assessment[0]}{alpha_text}</div>
        </div>
        
        <div class="card">
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value {'positive' if summary['total_return_pct'] > 0 else 'negative'}">{summary['total_return_pct']:+.2f}%</div>
                    <div class="metric-label">ML Strategy Return</div>
                </div>
                <div class="metric">
                    <div class="metric-value {spy_return_class}">{spy_return_str}</div>
                    <div class="metric-label">S&P 500 Return</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['sharpe_ratio']:.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric">
                    <div class="metric-value negative">{summary['max_drawdown_pct']:.2f}%</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['win_rate_pct']:.1f}%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['annualized_volatility_pct']:.1f}%</div>
                    <div class="metric-label">Volatility (Ann.)</div>
                </div>
            </div>
        </div>
        
        <h2>📊 Equity Curve</h2>
        <div class="card chart-container">
            <img src="equity_curve.png" alt="Equity Curve">
        </div>
        
        <div class="two-col">
            <div>
                <h2>📉 Drawdown</h2>
                <div class="card chart-container">
                    <img src="drawdown.png" alt="Drawdown">
                </div>
            </div>
            <div>
                <h2>📊 Returns Distribution</h2>
                <div class="card chart-container">
                    <img src="returns_distribution.png" alt="Returns Distribution">
                </div>
            </div>
        </div>
        
        <h2>📅 Monthly Returns</h2>
        <div class="card chart-container">
            <img src="monthly_returns.png" alt="Monthly Returns">
        </div>
        
        <div class="two-col">
            <div>
                <h2>🏆 Best Days</h2>
                <div class="card">
                    <table>
                        <tr><th>Date</th><th>Return</th></tr>
                        {"".join(f'<tr><td>{d}</td><td class="positive">{r*100:+.2f}%</td></tr>' for d, r in metrics['best_days'])}
                    </table>
                </div>
            </div>
            <div>
                <h2>📉 Worst Days</h2>
                <div class="card">
                    <table>
                        <tr><th>Date</th><th>Return</th></tr>
                        {"".join(f'<tr><td>{d}</td><td class="negative">{r*100:+.2f}%</td></tr>' for d, r in metrics['worst_days'])}
                    </table>
                </div>
            </div>
        </div>
        
        <h2>🎯 Top Positions</h2>
        <div class="card">
            <table>
                <tr><th>Symbol</th><th>Days Held</th><th>% of Period</th></tr>
                {"".join(f'<tr><td>{s}</td><td>{c}</td><td>{c/(summary["trading_days"]-1)*100:.1f}%</td></tr>' for s, c in metrics['top_positions'][:10])}
            </table>
        </div>
        
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Model trained on data before simulation period (no lookahead bias)</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, "w") as f:
        f.write(html)


def main():
    args = parse_args()
    
    # Validate args
    if args.start_date and not args.end_date:
        print("Error: --end-date is required when using --start-date")
        sys.exit(1)
    
    # Get results
    if args.from_json:
        print(f"Loading results from {args.from_json}...")
        with open(args.from_json) as f:
            results = json.load(f)
    else:
        print(f"Running simulation from {args.start_date} to {args.end_date}...")
        results = run_simulation(args.start_date, args.end_date, args.top_k)
    
    # Calculate additional metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(results)
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report name
    if args.report_name:
        base_name = args.report_name
    else:
        base_name = f"report_{results['summary']['start_date']}_to_{results['summary']['end_date']}"
    
    # Generate text summary
    print("Generating text summary...")
    text_summary = generate_text_summary(results, metrics)
    print(text_summary)
    
    # Save text summary
    text_path = output_dir / f"{base_name}.txt"
    with open(text_path, "w") as f:
        f.write(text_summary)
    print(f"\nText report saved to: {text_path}")
    
    # Save JSON results
    json_path = output_dir / f"{base_name}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON data saved to: {json_path}")
    
    # Generate charts
    if not args.no_charts:
        if not HAS_MATPLOTLIB:
            print("\nWarning: matplotlib not available, skipping charts")
        else:
            print("\nGenerating charts...")
            colors = setup_style(args.dark_mode)
            
            # Individual charts
            generate_equity_curve(results, metrics, colors, output_dir / "equity_curve.png")
            generate_drawdown_chart(results, metrics, colors, output_dir / "drawdown.png")
            generate_monthly_returns_heatmap(results, metrics, colors, output_dir / "monthly_returns.png")
            generate_returns_distribution(results, metrics, colors, output_dir / "returns_distribution.png")
            generate_position_frequency(results, metrics, colors, output_dir / "top_positions.png")
            
            # Summary dashboard
            generate_summary_dashboard(results, metrics, colors, output_dir / f"{base_name}_dashboard.png")
            
            print(f"Charts saved to: {output_dir}/")
            
            # Generate HTML report
            if args.html:
                html_path = output_dir / f"{base_name}.html"
                generate_html_report(results, metrics, output_dir, html_path)
                print(f"HTML report saved to: {html_path}")
    
    print("\n✅ Report generation complete!")


if __name__ == "__main__":
    main()

