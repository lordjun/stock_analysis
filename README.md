# A股板块龙头日报

这个项目每天通过 Akshare 读取 A 股收盘后的板块数据，默认找出全市场涨幅最高的 3 个细分概念板块、每个细分板块涨幅前 3 的龙头股，汇总相关新闻和关键词原因，并生成带 K 线图的 PPT 报告。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 运行

收盘后运行：

```powershell
python .\daily_sector_report.py
```

默认输出到 `reports` 目录。也可以指定日期和板块来源：

```powershell
python .\daily_sector_report.py --date 20260529 --data-source ths --board-source concept --output-dir reports
```

参数说明：

- `--date`：交易日期，格式 `YYYYMMDD`，默认今天。
- `--data-source`：板块数据源，`ths` 为同花顺，`em` 为东方财富，默认 `ths`。
- `--board-source`：`industry` 行业板块、`concept` 细分概念板块、`both` 两者合并后取前 3，默认 `concept`。
- `--lookback-days`：K 线图向前取多少天，默认 90。
- `--news-days`：原因分析新闻窗口，默认 14 天。

## 每天自动运行

Windows 任务计划示例，建议设置在交易日 16:30 后运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\register_daily_task.ps1
```

如果你的 Python 不在 PATH 中，可以打开 `scripts\run_daily_report.ps1`，把 `$Python` 改成你的 Python 绝对路径。

## 结果内容

PPT 包含：

- 当日全市场涨幅最高的 3 个细分概念板块。
- 每个细分板块中涨幅前 3 的龙头股。
- 每只龙头股近 90 个自然日左右的 K 线图。
- 基于新闻标题、板块/个股名称和关键词的上涨原因摘要。

说明：原因分析是自动化归纳，适合作为盘后复盘草稿，不构成投资建议。
