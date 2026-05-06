# OpenEvolve 量化因子演化复盘

## 一、项目背景与目标

### 问题定义

在A股市场发现一个截面IC均值 > 0.08、长期稳定的alpha因子。传统做法依赖量化研究员手动试错，周期长、维度有限。我们尝试用LLM驱动的进化算法（OpenEvolve框架）自动发现因子公式。

### 初始条件

- **数据**: 沪深300成分股，日线行情+基本面+技术指标，2021-2025
- **初始因子**: `bp = 1.0 / pb`（账面市值比），训练集IC=+0.044
- **评估方法**: 截面Spearman Rank IC，训练集2021-2023，验证集2024
- **进化框架**: OpenEvolve (MAP-Elites + 岛屿模型 + LLM变异)
- **LLM**: DeepSeek-Chat (70%) + DeepSeek-Reasoner (30%)

---

## 二、讨论与关键决策

### 决策1: 评估指标设计

**讨论**: 一开始使用简单的IC均值作为fitness，但发现存在两个问题：
1. 因子方向不确定——LLM生成的因子IC可能为负，需要人工判断取反
2. 训练集和验证集方向可能不一致，说明过拟合

**决定**: 设计多维度综合评分：
```
combined_score = 0.20 * |train_IC|/0.1 + 0.25 * |train_IR|/2.0
               + 0.10 * train_win_rate
               + 0.25 * |val_IC|/0.1 + 0.20 * |val_IR|/2.0
               - direction_penalty (train/val方向不一致时罚分)
```
同时加入auto-flip机制，检测到训练集IC为负时自动取反，让LLM只需关注因子逻辑。

**教训**: 这个设计非常成功。auto-flip + direction_penalty 的组合既保护了LLM探索的灵活性（允许方向错误），又通过罚分惩罚了过拟合。

### 决策2: MAP-Elites特征维度选择

**讨论**: 最初只用了2维（IC均值 + IC IR），但后来发现这不足以保持多样性——高IC的因子容易集中在同一格子里。

**决定**: 扩展到4维多样性空间：
- `abs_ic_mean`: |IC均值| — 预测能力
- `ic_ir`: |IC信息比| — 稳定性
- `ic_stability`: IC胜率 — 方向一致性
- `factor_turnover`: 因子换手率 — 交易成本

**教训**: 加入 `factor_turnover` 是关键。它确保了高换手（交易成本高）的因子和低换手的因子被分到不同格子，防止高频噪音因子淹没稳健低频因子。

### 决策3: 结构化诊断反馈

**讨论**: 最初的evaluator只返回数值指标，LLM需要自己解读。发现LLM经常：
- 不理解为什么score低（是IC弱？还是IR低？还是过拟合？）
- 不知道应该改什么（改因子类型？还是调参数？）

**决定**: 设计P0/P1/P2三级诊断系统：
- **P0（必须修复）**: 方向翻转依赖、训练/验证方向不一致
- **P1（应该优化）**: IC太弱(<0.02)、IR太低(<0.3)、验证集失效
- **P2（可以尝试）**: 亲子相关性低、覆盖率不足、胜率偏低

每个诊断附带结构化的参数调整建议和代码示例。

**教训**: 这是整个系统最大的杠杆点。诊断反馈质量直接决定了LLM的改进方向是否有效。P0/P1/P2的优先级排序确保了LLM先解决致命问题再追求锦上添花。

### 决策4: 知识库（RAG）集成

**讨论**: DeepSeek是通用LLM，对量化因子的领域知识有限。我们考虑了几个方案：
1. 直接在system prompt里写死因子知识 → 太长，且静态
2. 微调LLM → 成本高，且进化过程中产生的新知识无法利用
3. RAG知识库 → 灵活，可动态更新

**决定**: 建立40条种子知识 + 演化回写闭环的知识库系统：
- 两阶段检索: 规则过滤（problem_codes匹配）→ 向量相似度排序
- 知识回写: 当子代score相对父代提升>15%时，LLM自动总结改进经验写入KB
- 知识作为prompt中的"参考案例"而非"强制指令"

**教训**:
- 种子知识质量参差不齐，22/41条在本次演化中从未被检索到——规则过滤正确排除了不相关的知识
- 自生成知识（kb_4eb831ce8982）成为使用频率最高的条目（263次）——"演化中发现的知识比互联网种子更有针对性"
- 回写阈值15%偏严格，200轮只触发了1次回写

---

## 三、实现细节

### 3.1 整体架构

```
config.yaml ──> run_evolution.py ──> OpenEvolve.run_evolution()
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
            process_parallel.py    prompt/sampler.py      evaluator.py
            (并行进程管理)          (提示词构建)            (截面IC评估)
                    │                     │                     │
                    ▼                     ▼                     ▼
            knowledge_base/         LLM Ensemble           data_loader.py
            (RAG检索+回写)          (DeepSeek)             (Tushare缓存)
```

### 3.2 评估器核心实现

```python
class CrossSectionalICEvaluator:
    def evaluate_factor(self, compute_func) -> Dict:
        # 1. 对全股票池计算因子面板
        factor_panel = self.compute_factor_panel(compute_func)

        # 2. 截面Rank IC计算
        ic_series = self.calculate_cross_sectional_ic(factor_panel, method="spearman")

        # 3. Auto-flip: 训练集IC为负则自动翻转
        if train_ic.mean() < 0:
            factor_panel = -factor_panel; ic_series = -ic_series

        # 4. 训练集/验证集拆分 + 指标计算
        train_metrics = self._calc_ic_stats(train_ic, prefix="train")
        val_metrics = self._calc_ic_stats(val_ic, prefix="val")

        # 5. 综合评分
        combined_score = self._calc_combined_score(train_metrics, val_metrics)

        # 6. 结构化诊断
        diagnostics = self._generate_diagnostics(train, val, coverage, ...)

        # 7. MAP-Elites特征维度
        result["abs_ic_mean"] = abs(train_ic.mean())
        result["ic_ir"] = abs(train_ir)
        result["ic_stability"] = train_win_rate
        result["factor_turnover"] = self._calc_factor_turnover(factor_panel)
```

### 3.3 知识库集成实现

**检索流程**:
```
1. 从metrics提取diagnostic codes → ["IC_MODERATE", "IR_LOW"]
2. 从代码检测因子类别 → "value"
3. 从代码提取标签 → ["BP", "PB"]
4. 规则过滤: SQL WHERE problem_codes LIKE '%IC_MODERATE%' AND factor_category='value'
5. 构建查询文本 → 本地embedding server生成向量
6. 余弦相似度排序 → Top 5
7. 格式化为Markdown注入prompt
```

**回写流程**:
```
1. should_writeback(parent_score=0.350, child_score=0.520) → True (+48.6% > 15%)
2. LLM extract_knowledge(parent_code, child_code, metrics_diff) → 结构化JSON
3. writeback_to_kb() → 写入SQLite + 生成embedding + 更新缓存
```

### 3.4 数据加载实现

- 使用Tushare Pro API获取A股日线数据
- 三级缓存：单个股票parquet → 全量pickle → 内存
- 技术指标预计算（MA/EMA/MACD/RSI/KDJ/布林带等）存入缓存
- 缓存key基于 `md5(ts_code + start_date + end_date)`，支持不同时间范围的独立缓存

### 3.5 Prompt设计

**System Message 核心要素**:
1. 角色定义（量化因子研究员）
2. 可用数据字典（close/pb/pe/vol/turnover_rate等）
3. 因子设计原则（经济学逻辑 > 数据挖掘）
4. 经典因子思路（价值/动量/低波/质量）
5. 评估指标说明（IC/IR/胜率/换手率）

**User Message 核心要素**:
1. 当前因子表现表格（IC/IR/胜率/score，分训练集/验证集）
2. 结构化诊断结果（P0/P1/P2优先级问题列表）
3. 参数调整建议（带代码示例）
4. 知识库参考案例（Top-K相关改进经验）
5. SEARCH/REPLACE 代码修改格式

---

## 四、失败与教训

### 失败1: 小股票池导致过拟合

**现象**: 先用100只股票测试，训练集IC从0.044迅速跳到0.149，但验证集IC从0.062跌到0.020，score从0.42跌到0.30。

**根因**: 100只股票提供的信息量不足，LLM很快找到能"记住"训练集但泛化不了的噪音模式。同时早期scoring公式中训练集权重55%，奖励了过拟合。

**修复**: 扩展到全HS300（298只有效股票），验证集IC恢复正值。更大股票池天然提供了正则化效果。

**教训**: 截面因子的最小有效股票池应该≥200只。评分公式应给予验证集足够权重（最终设计25% val_IC + 20% val_IR = 45%样本外权重）。

### 失败2: Auto-flip掩盖方向错误

**现象**: 早期因子经常触发auto-flip，IC勉强为正但diagnostics一直提示"AUTO_FLIP"。

**根因**: LLM不理解某些因子的经济学方向（比如PB本身是反向指标——高PB=贵=低收益），写的因子方向反了。

**修复**: 在diagnostics中把AUTO_FLIP标记为P0问题，引导LLM在代码中显式修正方向。最终因子（BP=1/PB，low_vol_rank=1-vol）方向全正，无需auto-flip。

**教训**: Auto-flip是安全网，但不应成为常态。引导LLM理解因子经济学逻辑比机械翻转更重要。

### 失败3: 过度平滑导致信号滞后

**现象**: 中期版本（Gen 1-4）使用了多层EWM平滑（bp→ewm(10)→rank→ewm(5)→最终输出），IC_IR虽高但IC绝对值提升缓慢。

**根因**: 每层平滑都引入滞后。多层平滑叠加导致信号在时间上错位，削弱了截面预测能力。

**修复**: 最终版本（Gen 6）去掉了所有中间平滑层，直接 `bp_raw.rank(pct=True)`，只在最终组合后做一次 `rank(pct=True)` 标准化。

**教训**: 在截面因子中，"少即是多"。rank(pct=True)本身就是一种稳健的标准化，无需额外平滑。

### 失败4: backtrader框架不适用

**现象**: 用backtrader做300只股票的截面因子轮动回测，所有交易返回0收益。

**根因**: backtrader设计用于单标的策略回测（如CTA趋势跟踪），不适合多标的截面轮动。300个data feed的日期对齐和订单管理极其复杂。

**修复**: 用纯向量化方法重写回测——构建价格矩阵(date×stock)，在每日计算持仓市值，月初根据因子排名切换持仓。

**教训**: 工具选择要匹配问题特征。截面因子回测的核心是矩阵运算而非事件驱动。

### 失败5: 知识库种子知识冷启动

**现象**: 40条种子知识中22条从未被检索使用。第一条自生成知识反而成为最热门条目。

**根因**: 种子知识来自通用量化研究，覆盖momentum/quality/technical等方向，但本次演化LLM一直锁定在value方向。规则过滤正确排除了不相关方向的知识。

**修复**: 不需要修复——这是规则过滤设计正确的表现。但种子知识可以更聚焦于具体场景。

**教训**: 种子知识的价值不在于覆盖面广，而在于覆盖当前问题场景。更好的策略是"少而精"——20条高质量value/composite方向知识可能比40条泛化知识更有用。

---

## 五、最终结果

### 5.1 因子公式

```python
def compute_factor(data: pd.DataFrame) -> pd.Series:
    bp_raw = 1.0 / data['pb'].replace(0, np.nan)
    bp_rank = bp_raw.rank(pct=True)

    ep_raw = 1.0 / data['pe'].replace(0, np.nan)
    ep_rank = ep_raw.rank(pct=True)

    ret = data['close'].pct_change()
    vol = ret.ewm(span=20, min_periods=10).std().rank(pct=True)
    low_vol_rank = 1 - vol

    factor = 0.70 * bp_rank + 0.15 * ep_rank + 0.15 * low_vol_rank
    factor = factor.rank(pct=True)
    return factor
```

### 5.2 演化指标

| 阶段 | 迭代 | Score | Train IC | Train IR | Val IC | 关键变化 |
|------|------|-------|----------|----------|--------|---------|
| 种子 | 0 | 0.325 | 0.046 | 0.177 | 0.054 | BP单因子 |
| Gen 1 | 35 | 0.464 | 0.104 | 0.593 | 0.040 | +低波 +EWM平滑 |
| Gen 4 | 68 | 0.487 | 0.115 | 0.670 | 0.044 | 缩短窗口 +权重调整 |
| Gen 6 | 121 | **0.538** | **0.142** | **0.851** | **0.053** | +EP +去除中间平滑 |

### 5.3 10年回测 (2015-2025)

| 指标 | 数值 |
|------|------|
| 日均Rank IC | 0.0756 |
| 月度IC正率 | 81.8% (108/132) |
| IC信息比 | 0.426 |
| 最佳年份 | 2015: IC=0.148 |
| 最差年份 | 2020: IC=0.008 (成长抱团市) |

### 5.4 2026实盘回测 (1-4月)

| 月份 | 因子策略 | 等权基准 | 超额 |
|------|----------|----------|------|
| 2月 | +10.89% | +1.90% | +8.99% |
| 3月 | +5.52% | -6.99% | +12.51% |
| 累计 | **+21.59%** | -1.64% | **+23.23%** |

---

## 六、关键经验总结

1. **诊断反馈 > 原始指标**: P0/P1/P2三级诊断让LLM的改进方向准确率大幅提升
2. **知识库是放大器不是替代品**: 自生成知识比互联网种子更有价值，回写闭环至关重要
3. **截面因子=少即是多**: 3个子因子 + rank标准化，不需要复杂平滑
4. **大股票池=天然正则化**: ≥200只股票是截面因子有效演化的必要条件
5. **评分公式设计决定了演化方向**: 验证集权重≥45%才能避免过拟合
6. **向量化回测 > 事件驱动回测**: 截面因子轮动天然适合矩阵运算

---

## 七、可重用框架

本次经验已沉淀为通用框架，位于 `framework/` 目录，可直接复制迁移到其他 OpenEvolve 示例。

### 框架组件

| 模块 | 文件 | 功能 |
|------|------|------|
| DiagnosticsEngine | `framework/diagnostics.py` | P0/P1/P2 三级诊断，领域无关 |
| BaseEvaluator | `framework/base_evaluator.py` | 抽象评估器基类 + 评分配置 |
| VectorizedBacktester | `framework/backtest.py` | 向量化截面因子轮动回测 |
| PromptBuilder | `framework/prompt_builder.py` | 模板化提示词构建 + KB注入 |
| Config | `framework/config_schema.py` | 类型安全配置加载和校验 |
| KnowledgeBase | `knowledge_base/` | RAG 检索 + 回写闭环（独立模块）|

### 迁移路径

1. 复制 `framework/` 和 `knowledge_base/` 到目标项目
2. 继承 `BaseEvaluator` 实现 `evaluate_program()`
3. 配置 `config.yaml` 中 `knowledge_base` 部分
4. 准备领域 `seed_knowledge.json`
5. 参考 `framework/MIGRATION_GUIDE.md` 完成集成

详细迁移指南见 `framework/MIGRATION_GUIDE.md`。
