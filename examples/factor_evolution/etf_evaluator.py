"""
ETF策略评估器

6折交叉验证回测引擎：
- 2019-2024年数据，每年轮流做测试集
- 每个交易日调用策略函数，策略自主决定何时交易
- 支持滑动止损（策略自行实现）
- 适应度 = 平滑Calmar得分 + 一致性 + 收益
- 交易成本：单边 0.1%
- 快速预评估：先跑熊市年(2022)，过不了直接低分
"""

import os
import sys
import importlib.util
import logging
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

from etf_data_loader import ETFDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 配置 ──
COST_RATE = 0.001          # 单边交易成本 0.1%
MAX_POSITIONS = 10         # 最大持仓数量
YEARS = list(range(2019, 2025))  # 2019-2024共6年
N_FOLDS = 6               # 6折交叉验证
MIN_ETF_HISTORY = 120     # ETF至少有120天历史才可交易
LOOKBACK_LIMIT = 300      # 传给策略的最大历史天数（加速）

# 熊市年份（用于快速预评估）
HARD_YEARS = [2022, 2023]

# 核心可交易ETF池（2019年前已上市 + 高流动性行业ETF）
TRADEABLE_ETFS = [
    # 宽基
    "510300.SH", "510500.SH", "510050.SH", "159919.SZ", "159915.SZ",
    "588000.SH", "512100.SH",
    # 红利
    "515080.SH", "510880.SH", "512890.SH",
    # 金融
    "512800.SH", "512880.SH",
    # 消费
    "159928.SZ", "512690.SH",
    # 医药
    "512010.SH",
    # 科技
    "512480.SH", "515030.SH", "516160.SH", "515790.SH",
    # 军工
    "512660.SH",
    # 周期
    "515220.SH", "512400.SH",
    # 地产
    "512200.SH",
]


class ETFBacktester:
    """ETF策略回测引擎（带数据缓存加速）"""

    def __init__(self, etf_data: Dict[str, pd.DataFrame], northbound: Optional[pd.DataFrame] = None):
        self.etf_data = etf_data
        self.northbound = northbound
        self.all_dates = self._build_trading_calendar()

    def _build_trading_calendar(self) -> pd.DatetimeIndex:
        all_dates = set()
        for df in self.etf_data.values():
            all_dates.update(df.index)
        return pd.DatetimeIndex(sorted(all_dates))

    def run_backtest(
        self,
        strategy_func,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """
        执行回测（优化版：限制传给策略的历史深度）

        Returns:
            {'returns': pd.Series, 'metrics': dict, 'trades': int}
        """
        bt_start = pd.Timestamp(start_date)
        bt_end = pd.Timestamp(end_date)

        trading_dates = self.all_dates[(self.all_dates >= bt_start) & (self.all_dates <= bt_end)]
        if len(trading_dates) < 20:
            return {"returns": pd.Series(dtype=float), "metrics": {}, "trades": 0}

        available_etfs = {}
        for code, df in self.etf_data.items():
            history_before = df[df.index < bt_start]
            if len(history_before) >= MIN_ETF_HISTORY:
                available_etfs[code] = df

        if not available_etfs:
            return {"returns": pd.Series(dtype=float), "metrics": {}, "trades": 0}

        current_holdings = {}
        entry_prices = {}
        daily_returns = []
        n_trades = 0

        for i, date in enumerate(trading_dates):
            # 只传最近LOOKBACK_LIMIT天数据给策略（加速）
            etf_snapshot = {}
            for code, df in available_etfs.items():
                historical = df[df.index <= date]
                if len(historical) > 0:
                    etf_snapshot[code] = historical.iloc[-LOOKBACK_LIMIT:]

            try:
                target_weights = strategy_func(
                    etf_snapshot,
                    date,
                    current_holdings.copy(),
                    entry_prices.copy(),
                )
            except Exception:
                target_weights = current_holdings.copy()

            target_weights = self._validate_weights(target_weights, etf_snapshot)

            cost = self._calc_trade_cost(current_holdings, target_weights)
            if cost > 0:
                n_trades += 1

            for code in target_weights:
                if code not in current_holdings or current_holdings.get(code, 0) == 0:
                    if code in etf_snapshot and len(etf_snapshot[code]) > 0:
                        entry_prices[code] = etf_snapshot[code]["close"].iloc[-1]

            for code in list(entry_prices.keys()):
                if target_weights.get(code, 0) == 0:
                    entry_prices.pop(code, None)

            current_holdings = target_weights

            port_return = -cost
            if i < len(trading_dates) - 1:
                next_date = trading_dates[i + 1]
                for code, weight in current_holdings.items():
                    if weight > 0 and code in etf_snapshot:
                        df = available_etfs[code]
                        if date in df.index and next_date in df.index:
                            today_close = df.loc[date, "close"]
                            next_close = df.loc[next_date, "close"]
                            if today_close > 0:
                                ret = (next_close / today_close - 1) * weight
                                port_return += ret

            daily_returns.append(port_return)

        ret_series = pd.Series(daily_returns, index=trading_dates, name="portfolio")
        metrics = self._calc_metrics(ret_series)

        return {"returns": ret_series, "metrics": metrics, "trades": n_trades}

    def _validate_weights(self, weights: Any, etf_snapshot: Dict) -> Dict[str, float]:
        if not isinstance(weights, dict):
            return {}

        valid = {}
        for code, w in weights.items():
            if not isinstance(w, (int, float)) or np.isnan(w) or w <= 0:
                continue
            if code not in etf_snapshot:
                continue
            valid[code] = float(w)

        if len(valid) > MAX_POSITIONS:
            sorted_items = sorted(valid.items(), key=lambda x: x[1], reverse=True)
            valid = dict(sorted_items[:MAX_POSITIONS])

        total = sum(valid.values())
        if total > 1.0:
            for code in valid:
                valid[code] /= total

        return valid

    def _calc_trade_cost(self, old_weights: Dict, new_weights: Dict) -> float:
        all_codes = set(list(old_weights.keys()) + list(new_weights.keys()))
        turnover = 0.0
        for code in all_codes:
            old_w = old_weights.get(code, 0)
            new_w = new_weights.get(code, 0)
            turnover += abs(new_w - old_w)
        return turnover * COST_RATE

    @staticmethod
    def _calc_metrics(ret_series: pd.Series) -> Dict[str, float]:
        if len(ret_series) < 10:
            return {"calmar": 0, "annual_return": 0, "max_drawdown": -1, "sharpe": 0, "win_rate": 0}

        cum_ret = (1 + ret_series).cumprod()
        total_ret = cum_ret.iloc[-1] - 1
        n_days = len(ret_series)
        ann_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1

        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()

        ann_vol = ret_series.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        calmar = ann_ret / abs(max_dd) if max_dd < 0 else ann_ret * 10

        win_rate = (ret_series > 0).sum() / len(ret_series) if len(ret_series) > 0 else 0

        return {
            "calmar": calmar,
            "annual_return": ann_ret,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "total_return": total_ret,
            "trading_days": n_days,
        }


class ETFStrategyEvaluator:
    """ETF策略评估器 — OpenEvolve 接口"""

    def __init__(self):
        self.etf_data = None
        self.northbound = None
        self.backtester = None
        self._load_data()

    def _load_data(self):
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file, override=True)

        token = os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            raise ValueError("缺少 TUSHARE_TOKEN")

        loader = ETFDataLoader(token=token)

        logger.info("加载ETF数据...")
        self.etf_data = {}
        for code in TRADEABLE_ETFS:
            df = loader.load_etf_data(code, "20180101", "20260101")
            if df is not None and len(df) > 60:
                self.etf_data[code] = df

        logger.info(f"已加载 {len(self.etf_data)} 只ETF数据")

        self.northbound = loader.load_northbound_flow("20180101", "20260101")
        self.backtester = ETFBacktester(self.etf_data, self.northbound)

    def _get_cv_folds(self) -> List[Tuple[int, str, str]]:
        """6折交叉验证：2019-2024每年轮做测试集"""
        folds = []
        for test_year in YEARS:
            test_start = f"{test_year}0101"
            test_end = f"{test_year}1231"
            folds.append((test_year, test_start, test_end))
        return folds

    def evaluate(self, program_path: str) -> Dict[str, Any]:
        """评估策略（带快速预评估）"""
        try:
            strategy_func = self._load_strategy(program_path)
        except Exception as e:
            logger.error(f"加载策略失败: {e}")
            return self._fail_result(f"加载失败: {e}")

        if not self._sanity_check(strategy_func):
            return self._fail_result("策略健全性检查失败")

        # 快速预评估：先跑最难的年份
        for hard_year in HARD_YEARS:
            try:
                r = self.backtester.run_backtest(
                    strategy_func, f"{hard_year}0101", f"{hard_year}1231"
                )
                m = r["metrics"]
                if m.get("max_drawdown", -1) < -0.30:
                    logger.info(f"快速预评估: {hard_year}年回撤{m['max_drawdown']:.1%}过大，提前终止")
                    return self._early_fail_result(hard_year, m)
            except Exception:
                pass

        # 完整6折交叉验证
        folds = self._get_cv_folds()
        fold_results = []

        for test_year, test_start, test_end in folds:
            try:
                result = self.backtester.run_backtest(strategy_func, test_start, test_end)
                metrics = result["metrics"]
                fold_results.append({
                    "year": test_year,
                    "annual_return": metrics.get("annual_return", 0),
                    "max_drawdown": metrics.get("max_drawdown", -1),
                    "calmar": metrics.get("calmar", 0),
                    "sharpe": metrics.get("sharpe", 0),
                    "trades": result.get("trades", 0),
                })
            except Exception as e:
                logger.warning(f"Fold {test_year} 回测异常: {e}")
                fold_results.append({
                    "year": test_year,
                    "annual_return": -0.5,
                    "max_drawdown": -0.5,
                    "calmar": 0,
                    "sharpe": 0,
                    "trades": 0,
                })

        score, score_details = self._calc_fitness(fold_results)

        return {
            "score": score,
            "metrics": score_details,
            "fold_results": fold_results,
        }

    def _calc_fitness(self, fold_results: List[Dict]) -> Tuple[float, Dict]:
        """
        平滑适应度得分（0~1区间，有区分度）

        设计原则：
        - 差策略也能得到非零分（便于演化搜索）
        - 好策略有明确的提升方向
        - 风控违规用连续惩罚代替阶梯惩罚
        """
        annual_returns = [f["annual_return"] for f in fold_results]
        max_drawdowns = [f["max_drawdown"] for f in fold_results]
        calmars = [f["calmar"] for f in fold_results]
        trades = [f["trades"] for f in fold_results]

        avg_calmar = np.mean(calmars)
        worst_return = min(annual_returns)
        worst_dd = min(max_drawdowns)
        avg_return = np.mean(annual_returns)
        avg_trades = np.mean(trades)

        ret_std = np.std(annual_returns)
        consistency = 1.0 / (1.0 + ret_std * 5)

        profitable_folds = sum(1 for r in annual_returns if r > 0)
        n_folds = len(annual_returns)

        # ── 基础存活分（0.05）── 只要策略能跑就给
        score = 0.05

        # ── 收益贡献（0~0.30）── sigmoid映射，平均年化25%→满分
        if avg_return > 0:
            ret_score = 0.30 * min(avg_return / 0.25, 1.0)
        else:
            ret_score = 0.30 * max(avg_return / 0.25, -0.5)
        score += ret_score

        # ── Calmar贡献（0~0.25）── 平均Calmar 2.0→满分
        calmar_score = 0.25 * min(max(avg_calmar, 0) / 2.0, 1.0)
        score += calmar_score

        # ── 一致性贡献（0~0.15）──
        score += consistency * 0.15

        # ── 盈利fold比例（0~0.15）──
        score += (profitable_folds / n_folds) * 0.15

        # ── 交易活跃度（0~0.10）── 至少每年10次交易
        if avg_trades >= 10:
            trade_score = 0.10
        elif avg_trades >= 3:
            trade_score = 0.10 * (avg_trades / 10.0)
        else:
            trade_score = 0.0
        score += trade_score

        # ── 连续惩罚（代替阶梯式扣分）──
        # 亏损年惩罚：每个亏损fold按亏损幅度线性扣分
        loss_penalty = 0.0
        for r in annual_returns:
            if r < 0:
                loss_penalty += min(abs(r) * 0.5, 0.08)
        score -= loss_penalty

        # 回撤惩罚：超过15%的回撤按幅度扣分
        dd_penalty = 0.0
        for dd in max_drawdowns:
            if dd < -0.15:
                dd_penalty += min((abs(dd) - 0.15) * 0.8, 0.10)
        score -= dd_penalty

        # 不活跃惩罚
        if avg_trades < 2:
            score -= 0.15

        score = max(score, 0.0)
        score = min(score, 1.0)

        details = {
            "avg_calmar": avg_calmar,
            "avg_return": avg_return,
            "worst_year_return": worst_return,
            "worst_drawdown": worst_dd,
            "consistency": consistency,
            "avg_trades": avg_trades,
            "profitable_folds": profitable_folds,
            "loss_penalty": loss_penalty,
            "dd_penalty": dd_penalty,
        }

        return score, details

    def _load_strategy(self, program_path: str):
        spec = importlib.util.spec_from_file_location("strategy_module", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "compute_strategy"):
            raise ValueError("策略文件缺少 compute_strategy 函数")
        return module.compute_strategy

    def _sanity_check(self, strategy_func) -> bool:
        try:
            code = list(self.etf_data.keys())[0]
            sample_data = {code: self.etf_data[code].iloc[:30]}
            date = self.etf_data[code].index[29]

            result = strategy_func(sample_data, date, {}, {})
            if result is None:
                return False
            if not isinstance(result, dict):
                return False
            return True
        except Exception as e:
            logger.warning(f"健全性检查失败: {e}")
            return False

    @staticmethod
    def _fail_result(reason: str) -> Dict:
        return {
            "score": 0.0,
            "metrics": {
                "error": reason, "avg_calmar": 0, "worst_year_return": -1,
                "avg_trades": 0, "consistency": 0, "avg_return": 0,
                "worst_drawdown": -1, "profitable_folds": 0,
                "loss_penalty": 0, "dd_penalty": 0,
            },
            "fold_results": [],
        }

    @staticmethod
    def _early_fail_result(year: int, metrics: Dict) -> Dict:
        """快速预评估失败 — 给一个低但非零的分数"""
        return {
            "score": 0.02,
            "metrics": {
                "error": f"{year}年回撤过大({metrics.get('max_drawdown', -1):.1%})",
                "avg_calmar": 0, "worst_year_return": metrics.get("annual_return", -0.5),
                "avg_trades": 0, "consistency": 0, "avg_return": 0,
                "worst_drawdown": metrics.get("max_drawdown", -1),
                "profitable_folds": 0, "loss_penalty": 0, "dd_penalty": 0,
            },
            "fold_results": [],
        }


# ── OpenEvolve 入口 ──────────────────────────────────────────

_evaluator = None


def evaluate(program_path: str, **kwargs) -> Dict[str, Any]:
    """OpenEvolve 评估入口函数（同步）"""
    global _evaluator
    if _evaluator is None:
        _evaluator = ETFStrategyEvaluator()

    result = _evaluator.evaluate(program_path)

    score = result["score"]
    metrics = result.get("metrics", {})
    logger.info(
        f"策略评分: {score:.4f} | "
        f"平均Calmar: {metrics.get('avg_calmar', 0):.2f} | "
        f"平均年化: {metrics.get('avg_return', 0):.1%} | "
        f"最差年: {metrics.get('worst_year_return', 0):.1%} | "
        f"盈利folds: {metrics.get('profitable_folds', 0)}/{N_FOLDS}"
    )

    return {
        "combined_score": score,
        "avg_calmar": metrics.get("avg_calmar", 0),
        "worst_year_return": metrics.get("worst_year_return", -1),
        "avg_trades_per_year": metrics.get("avg_trades", 0),
        "consistency": metrics.get("consistency", 0),
        "avg_return": metrics.get("avg_return", 0),
        "worst_drawdown": metrics.get("worst_drawdown", -1),
        "profitable_folds": metrics.get("profitable_folds", 0),
    }


# ── 命令行测试 ──────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python etf_evaluator.py <strategy_file.py>")
        print("  示例: python etf_evaluator.py initial_strategy.py")
        sys.exit(1)

    strategy_path = sys.argv[1]
    if not os.path.exists(strategy_path):
        print(f"文件不存在: {strategy_path}")
        sys.exit(1)

    result = evaluate(strategy_path)
    print(f"\n{'='*60}")
    print(f"综合评分: {result['combined_score']:.4f}")
    print(f"平均Calmar: {result['avg_calmar']:.2f}")
    print(f"平均年化: {result['avg_return']:.1%}")
    print(f"最差年收益: {result['worst_year_return']:.1%}")
    print(f"盈利年份: {result['profitable_folds']}/{N_FOLDS}")
    print(f"{'='*60}")

    evaluator = _evaluator
    if evaluator:
        folds = evaluator._get_cv_folds()
        strategy_func = evaluator._load_strategy(strategy_path)
        print(f"\n{'年份':<6} {'年化收益':<10} {'最大回撤':<10} {'Calmar':<8} {'交易次数':<8}")
        print("-" * 50)
        for test_year, test_start, test_end in folds:
            r = evaluator.backtester.run_backtest(strategy_func, test_start, test_end)
            m = r["metrics"]
            print(
                f"{test_year:<6} "
                f"{m.get('annual_return',0):>+8.1%}  "
                f"{m.get('max_drawdown',0):>+8.1%}  "
                f"{m.get('calmar',0):>6.2f}  "
                f"{r.get('trades',0):>6}"
            )
