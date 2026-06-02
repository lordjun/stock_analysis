# Darvas Weekly Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible A-share weekly Darvas breakout scanner.

**Architecture:** Add one standalone Python module that can be imported for tests and executed as a CLI. Keep weekly aggregation, signal generation, exit simulation, and reporting as separate functions.

**Tech Stack:** Python, pandas, numpy, AKShare, unittest.

---

### Task 1: Lock Core Backtest Behavior With Tests

**Files:**
- Create: `tests/test_darvas_weekly_backtest.py`

- [ ] Write tests for breakout detection, fixed stop, trailing drawdown stop, and five-week time exit using synthetic weekly bars.
- [ ] Run `python -m unittest tests.test_darvas_weekly_backtest -v` and confirm it fails because `darvas_weekly_backtest` does not exist.

### Task 2: Implement Importable Backtest Core

**Files:**
- Create: `darvas_weekly_backtest.py`

- [ ] Implement `aggregate_weekly`, `find_signals`, `simulate_trade`, and summary helpers.
- [ ] Run `python -m unittest tests.test_darvas_weekly_backtest -v` and confirm tests pass.

### Task 3: Add Data Fetching And CLI

**Files:**
- Modify: `darvas_weekly_backtest.py`

- [ ] Implement AKShare stock universe loading.
- [ ] Implement per-stock historical fetch with forward adjustment.
- [ ] Add CLI options for years, volume multiplier, box weeks, hold weeks, stop, drawdown, output directory, and optional limit.
- [ ] Write CSV and Markdown reports.

### Task 4: Verify And Run

**Files:**
- Generated: `reports/darvas/darvas_weekly_trades_*.csv`
- Generated: `reports/darvas/darvas_weekly_summary_*.md`

- [ ] Run unit tests.
- [ ] Run a small limited scan to verify upstream data and report generation.
- [ ] Run the full A-share five-year scan.
- [ ] Report win rate and key caveats.
