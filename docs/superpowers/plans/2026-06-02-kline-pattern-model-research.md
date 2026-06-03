# K-line Pattern Model Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible research workflow to find and validate a low-frequency A-share K-line pattern model with win rate above 50%, strict stops, and positive expectancy.

**Architecture:** Start from the existing `darvas_weekly_backtest.py` and extract reusable research units only when they are needed: data access, signal rules, trade simulation, metrics, and reporting. Candidate models are compared under one shared backtest harness so their results use the same universe, dates, costs, and validation splits.

**Tech Stack:** Python 3.12, pandas, numpy, AKShare, unittest, Markdown/CSV reports, PortableGit at `C:\Users\Lordjun\Documents\Git\tools\PortableGit\cmd\git.exe`.

---

## File Structure

- `agent.md`: already saved project instructions and execution gates.
- `docs/superpowers/plans/2026-06-02-kline-pattern-model-research.md`: this plan.
- `darvas_weekly_backtest.py`: existing Darvas weekly breakout backtest; first smoke-test target and possible source for shared functions.
- `tests/test_darvas_weekly_backtest.py`: existing unit tests for current backtest logic.
- `kline_model_research.py`: create only after baseline and environment checks; shared research runner for candidate models.
- `tests/test_kline_model_research.py`: create before implementing shared model logic.
- `reports/kline_models/`: generated Markdown/CSV outputs for model comparisons.
- `data/cache/kline_models/`: optional generated local cache if online data is stable enough to cache. Do not commit generated cache files.

## Task 1: Git Baseline And Repository Hygiene

**Files:**
- Read: `.gitignore`
- Read: `agent.md`
- Read: `docs/superpowers/plans/2026-06-02-kline-pattern-model-research.md`

- [ ] **Step 1: Verify PortableGit works**

Run:

```powershell
& 'C:\Users\Lordjun\Documents\Git\tools\PortableGit\cmd\git.exe' --version
```

Expected: prints `git version 2.54.0.windows.1` or another valid Git version.

- [ ] **Step 2: Verify repository status**

Run:

```powershell
& 'C:\Users\Lordjun\Documents\Git\tools\PortableGit\cmd\git.exe' -c core.quotepath=false status --short
```

Expected: current project files are untracked in a newly initialized repository.

- [ ] **Step 3: Check Git author config**

Run:

```powershell
& 'C:\Users\Lordjun\Documents\Git\tools\PortableGit\cmd\git.exe' config user.name
& 'C:\Users\Lordjun\Documents\Git\tools\PortableGit\cmd\git.exe' config user.email
```

Expected: both values are present. If either is missing, stop and ask the user which name/email to use, or ask them to configure Git manually.

- [ ] **Step 4: Ensure generated files are ignored**

Open `.gitignore`. It must ignore at least:

```gitignore
__pycache__/
*.pyc
reports/
data/cache/
```

If missing, add only the missing ignore rules with `apply_patch`.

- [ ] **Step 5: Create baseline commit**

Run:

```powershell
& 'C:\Users\Lordjun\Documents\Git\tools\PortableGit\cmd\git.exe' add .gitignore README.md requirements.txt agent.md daily_sector_report.py darvas_weekly_backtest.py scripts tests docs inputs
& 'C:\Users\Lordjun\Documents\Git\tools\PortableGit\cmd\git.exe' commit -m "chore: baseline stock analysis project"
```

Expected: one baseline commit is created. If commit fails because Git cannot write locks under `.git`, stop and report the exact command for the user to run manually.

## Task 2: Existing Environment Smoke Test

**Files:**
- Read: `requirements.txt`
- Read: `darvas_weekly_backtest.py`
- Read: `tests/test_darvas_weekly_backtest.py`

- [ ] **Step 1: Check Python version**

Run:

```powershell
python --version
```

Expected: Python 3.10+; Python 3.12 is preferred because this project already uses it.

- [ ] **Step 2: Check required package imports**

Run:

```powershell
python -c "import pandas, numpy, akshare, requests, matplotlib, mplfinance; print('imports ok')"
```

Expected: `imports ok`. If a package is missing, do not install immediately; report the missing package and ask for approval before installing.

- [ ] **Step 3: Run existing unit tests**

Run:

```powershell
python -m unittest tests.test_darvas_weekly_backtest -v
```

Expected: all current tests pass. If they fail, use systematic debugging before changing behavior.

- [ ] **Step 4: Compile existing scripts**

Run:

```powershell
python -m py_compile .\daily_sector_report.py .\darvas_weekly_backtest.py
```

Expected: no syntax errors.

## Task 3: Data Source Smoke Test

**Files:**
- Read: `darvas_weekly_backtest.py`
- Generated only if needed: `reports/kline_models/data_smoke_YYYYMMDD.md`

- [ ] **Step 1: Test AKShare stock universe**

Run:

```powershell
python -c "import akshare as ak; df=ak.stock_info_a_code_name(); print(df.head().to_string(index=False)); print(len(df))"
```

Expected: a non-empty A-share code/name table.

- [ ] **Step 2: Test one stable daily history pull**

Run:

```powershell
python -c "import akshare as ak; df=ak.stock_zh_a_hist(symbol='000001', period='daily', start_date='20160101', end_date='20260602', adjust='qfq'); print(df.tail().to_string(index=False)); print(df.shape)"
```

Expected: non-empty daily bars for `000001`.

- [ ] **Step 3: Test small existing Darvas scan**

Run:

```powershell
python .\darvas_weekly_backtest.py --years 3 --limit 20 --output-dir reports/darvas_smoke
```

Expected: CSV and Markdown outputs are generated under `reports/darvas_smoke/`. If online data is unstable, record failure rate and consider alternate provider before large scans.

## Task 4: Define Shared Research Core With Tests

**Files:**
- Create: `tests/test_kline_model_research.py`
- Create: `kline_model_research.py`

- [ ] **Step 1: Write tests for core trade simulation**

Create tests covering:

```python
import unittest
import pandas as pd

from kline_model_research import ExitRule, TradeSetup, simulate_position


def bar(date, open_, high, low, close, volume=1000):
    return {
        "date": pd.Timestamp(date),
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume),
    }


class KlineModelResearchTests(unittest.TestCase):
    def test_initial_stop_exits_before_time_exit(self):
        future = pd.DataFrame([
            bar("2025-01-06", 100, 104, 97, 102),
            bar("2025-01-07", 102, 103, 89, 91),
        ])
        setup = TradeSetup(
            code="000001",
            name="Ping An Bank",
            signal_date=pd.Timestamp("2025-01-03"),
            entry_date=pd.Timestamp("2025-01-06"),
            entry_price=100.0,
            initial_stop=92.0,
            max_holding_bars=20,
            exit_rule=ExitRule.TRAILING_HIGH_DRAWDOWN,
            trailing_drawdown=0.18,
        )
        trade = simulate_position(setup, future)
        self.assertEqual("initial_stop", trade.exit_reason)
        self.assertAlmostEqual(-0.08, trade.realized_return)

    def test_trailing_stop_lets_profit_run_then_exits_positive(self):
        future = pd.DataFrame([
            bar("2025-01-06", 100, 125, 99, 122),
            bar("2025-01-07", 122, 126, 102, 105),
        ])
        setup = TradeSetup(
            code="000001",
            name="Ping An Bank",
            signal_date=pd.Timestamp("2025-01-03"),
            entry_date=pd.Timestamp("2025-01-06"),
            entry_price=100.0,
            initial_stop=92.0,
            max_holding_bars=20,
            exit_rule=ExitRule.TRAILING_HIGH_DRAWDOWN,
            trailing_drawdown=0.18,
        )
        trade = simulate_position(setup, future)
        self.assertEqual("trailing_stop", trade.exit_reason)
        self.assertAlmostEqual(0.0250, trade.realized_return, places=4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```powershell
python -m unittest tests.test_kline_model_research -v
```

Expected: fails because `kline_model_research` does not exist.

- [ ] **Step 3: Implement minimal shared simulation core**

Create `TradeSetup`, `ExitRule`, `CompletedTrade`, and `simulate_position` in `kline_model_research.py`. The implementation must:

- Use `entry_price`.
- Apply `initial_stop`.
- Track highest high after entry.
- Apply `trailing_drawdown` from the highest high confirmed before the current bar. Do not assume the current bar's high occurred before its low.
- Exit by max holding bars when no stop is hit.
- Return realized return and exit reason.

- [ ] **Step 4: Run tests and confirm they pass**

Run:

```powershell
python -m unittest tests.test_kline_model_research -v
```

Expected: all tests pass.

## Task 5: Implement Candidate Model Comparison

**Files:**
- Modify: `kline_model_research.py`
- Modify: `tests/test_kline_model_research.py`
- Generated: `reports/kline_models/`

- [ ] **Step 1: Add candidate model tests**

Add synthetic-bar tests for:

- Weekly box breakout requires prior-box high breakout, volume confirmation, and trend filter.
- Trend pullback restart requires bullish moving-average alignment, pullback near the selected average, and close back above trigger.
- Strong consolidation breakout requires prior uptrend, volatility contraction, and breakout above consolidation high.

- [ ] **Step 2: Implement candidate signal functions**

Implement functions with explicit parameters:

```python
find_weekly_box_breakouts(...)
find_trend_pullback_restarts(...)
find_consolidation_breakouts(...)
```

Each function returns signal objects that can be converted to `TradeSetup`.

- [ ] **Step 3: Add metrics aggregation**

Implement metrics:

- trade count
- win rate
- average return
- median return
- average winner
- average loser
- profit/loss ratio
- expectancy
- max drawdown from equity curve
- annual win rate
- exit reason counts
- holding-period distribution

- [ ] **Step 4: Add CLI runner**

CLI options must include:

```text
--start-date
--end-date
--universe
--limit
--model
--output-dir
--fee-rate
--slippage-rate
--max-workers
```

Default output directory: `reports/kline_models`.

## Task 6: Run Research In Expanding Stages

**Files:**
- Generated: `reports/kline_models/*.csv`
- Generated: `reports/kline_models/*.md`

- [ ] **Step 1: Small run**

Run a small universe or `--limit 30` test over 5 years.

Expected: all candidate models produce outputs or clear no-signal reports.

- [ ] **Step 2: Medium run**

Run `--limit 300` over 10 years.

Expected: enough trades to identify whether models are promising before full-universe execution.

- [ ] **Step 3: Full or broad run**

Run full A-share or a reproducible broad universe over 10 years only after the medium run is stable.

Expected: final model comparison report with data failures listed separately.

- [ ] **Step 4: Validation split**

Report results for:

- Train: 2016-01-01 to 2021-12-31.
- Validation: 2022-01-01 to 2023-12-31.
- Out-of-sample test: 2024-01-01 to 2026-06-02.

If data availability forces different dates, state the exact dates used.

## Task 7: Final Research Report

**Files:**
- Create: `reports/kline_models/final_kline_model_research_YYYYMMDD.md`

- [ ] **Step 1: Compare candidate models**

Include one table per model:

- total trades
- win rate
- average profit
- average loss
- profit/loss ratio
- expectancy
- max drawdown
- annual result range
- parameter stability result

- [ ] **Step 2: Select or reject model**

If a model passes acceptance, document:

- entry rule
- stop rule
- trailing exit rule
- max holding period
- position-risk note
- suitable market environment
- unsuitable market environment

If no model passes, document the failure evidence and next research direction.

- [ ] **Step 3: Final verification**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests pass.

## Plan Self-Review

- Spec coverage: covers `agent.md` execution gates, 10-year A-share data, low-frequency holding period, strict stops, positive expectancy, data-source fallback, no unapproved dependency installation, and final report requirements.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation gates remain.
- Type consistency: planned `TradeSetup`, `ExitRule`, `CompletedTrade`, and `simulate_position` are introduced before later tasks reference them.

## Execution Notes

- PortableGit path verified through explicit `--git-dir` and `--work-tree` usage because the sandbox cannot write `.git` directly.
- `agent.md` was saved after user approval.
- `kline_model_research.py` and `tests/test_kline_model_research.py` were created with a shared weekly-bar research harness, candidate signal functions, local cache fallback, transaction-cost adjustment, and sweep mode.
- Unit coverage currently includes signal dispatch, default parameter consistency, no same-bar trailing-stop lookahead, non-overlapping trades, AKShare normalization, weekly aggregation, nonpositive OHLC filtering, and sweep ranking.
- Pure trailing-stop trend/breakout models did not meet the >50% win-rate requirement on the cached 100-stock sample; best broad sweep win rate was about 29.14%.
- Strong-trend filtering improved expectancy but not win rate. The closest trend-running versions stayed below 50% win rate.
- A candidate using strong-trend pullback restart plus fixed +15% profit target and -12% stop loss passed the hard win-rate/expectancy requirement on the first 200 non-ST cached stocks:
  - 222 non-overlapping trades
  - 51.35% win rate
  - 1.61% average return per trade after estimated costs
  - 14.46% average winner
  - -11.95% average loser
  - 1.21 profit/loss ratio
- Candidate report: `reports/kline_models_candidate_10y_limit200/summary.md`.
- Remaining validation: broaden beyond 200 stocks and add train/validation/test date splits before treating the model as robust.
