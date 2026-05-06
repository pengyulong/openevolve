# 框架迁移指南

本文档说明如何将 `framework/` 和 `knowledge_base/` 模块迁移到其他 OpenEvolve 示例项目中。

## 目录结构

迁移后的目标项目结构：

```
your_example/
├── config.yaml                  # 项目配置（含 knowledge_base 部分）
├── your_initial_program.py      # 初始程序
├── your_evaluator.py            # 评估器（继承 BaseEvaluator）
├── your_data_loader.py          # 数据加载器
├── run_evolution.py             # 进化入口
├── prompts/                     # 提示词模板
│   ├── system_message.txt
│   └── user_message.txt
├── knowledge_base/              # 知识库模块（直接复制）
│   ├── __init__.py
│   ├── kb_manager.py
│   ├── kb_embedder.py
│   ├── kb_retriever.py
│   ├── kb_writer.py
│   ├── init_kb.py
│   ├── kb_schema.sql
│   └── seed_knowledge.json       # 领域种子知识
├── framework/                   # 通用框架（直接复制）
│   ├── __init__.py
│   ├── diagnostics.py
│   ├── base_evaluator.py
│   ├── backtest.py
│   ├── config_schema.py
│   └── prompt_builder.py
└── data_cache/                  # 数据缓存目录
```

## 迁移步骤

### Step 1: 复制框架模块

```bash
# 从 factor_evolution 复制
cp -r examples/factor_evolution/framework/ your_example/framework/
cp -r examples/factor_evolution/knowledge_base/ your_example/knowledge_base/
```

### Step 2: 实现自定义评估器

继承 `BaseEvaluator` 实现领域特定的评估逻辑：

```python
# your_evaluator.py
from framework.base_evaluator import BaseEvaluator, ScoringConfig, create_evaluate_function

class MyEvaluator(BaseEvaluator):
    def __init__(self, data, config):
        # 配置评分权重
        scoring = ScoringConfig(
            metrics={
                "train_score": {"weight": 0.5, "target": 1.0},
                "val_score":   {"weight": 0.5, "target": 1.0},
            },
            penalties=[
                {"name": "overfit", "weight": 0.2,
                 "condition": "train_score > 2 * val_score"},
            ],
        )
        super().__init__(scoring=scoring)

        # 注册自定义诊断规则
        self.diagnostics.register_rule(self._my_custom_check)

        self.data = data
        self.config = config

    def evaluate_program(self, program_path: str) -> Dict:
        module = self.load_program(program_path)

        # 1. 执行评估
        train_score = ...  # 训练集得分
        val_score = ...     # 验证集得分

        # 2. 综合评分
        metrics = {
            "train_score": train_score,
            "val_score": val_score,
            "combined_score": 0.0,
        }
        metrics["combined_score"] = self.calc_combined_score(metrics)

        # 3. 添加诊断信息
        diagnostics = self.format_diagnostics(metrics)
        metrics.update(diagnostics)

        # 4. 添加 MAP-Elites 特征维度
        metrics["feature_1"] = train_score
        metrics["feature_2"] = val_score

        return metrics

    def _my_custom_check(self, metrics: Dict) -> List:
        """自定义诊断规则"""
        # 返回 DiagnosticIssue 列表
        ...

# 创建 OpenEvolve evaluate() 入口
def make_evaluator():
    config = load_config("config.yaml")
    data = load_my_data(config)
    return MyEvaluator(data, config)

evaluate = create_evaluate_function(make_evaluator)
```

### Step 3: 配置知识库

在 `config.yaml` 中添加：

```yaml
knowledge_base:
  enabled: true
  db_path: "knowledge_base/kb_store.db"
  seed_data_path: "knowledge_base/seed_knowledge.json"
  embedding_local_url: "http://your-server:8190/llm_service/embedding"  # 可选
  retrieval_top_k: 5
  writeback_threshold: 0.15
```

如果没有 embedding server，知识库会使用哈希 fallback，仍然可用但检索精度会降低。

### Step 4: 初始化知识库

```bash
cd your_example
python knowledge_base/init_kb.py
```

这将：
1. 创建 SQLite 数据库
2. 导入种子知识（从 seed_knowledge.json）
3. 生成嵌入向量缓存

### Step 5: 准备种子知识

编辑 `knowledge_base/seed_knowledge.json`，写入领域特定的改进经验。每条知识格式：

```json
{
  "id": "seed_001",
  "problem_codes": ["WEAK_SIGNAL"],
  "factor_category": "value",
  "tags": ["BP", "价值因子"],
  "context_before": "因子IC=0.02，信号偏弱",
  "improvement_action": "将BP与EP复合，权重7:3",
  "improvement_result": "IC提升至0.08，IR从0.3提升至0.6",
  "code_example": "factor = 0.7*bp_rank + 0.3*ep_rank",
  "search_text": "WEAK_SIGNAL value BP EP 价值因子 复合增强...",
  "success_rating": 0.8,
  "source": "seed",
  "status": "active"
}
```

关键字段：
- `problem_codes`: 必须与诊断引擎输出的问题码匹配
- `factor_category`: 与检索器的类别检测结果匹配
- `search_text`: 向量检索的匹配文本

### Step 6: 设计提示词模板

创建 `prompts/system_message.txt` 和 `prompts/user_message.txt`，使用 `{variable}` 语法：

```
# user_message.txt 示例

## 当前程序
```python
{current_program}
```

## 评估指标
| 指标 | 训练集 | 验证集 |
|------|--------|--------|
...

## 诊断反馈
{diagnostics_prio}

## 参数建议
{parameter_suggestions}

## 知识库参考
{knowledge_base_context}

## 任务
请改进上述程序...
```

### Step 7: 集成知识库到 OpenEvolve worker

在 worker 进程中初始化知识库检索器：

```python
# 在 process_parallel.py 的 worker 初始化中
from knowledge_base import KnowledgeBase, KBEmbedder, KnowledgeRetriever, KnowledgeWriter

def _init_worker_kb():
    kb = KnowledgeBase(db_path=config.knowledge_base.db_path)
    embedder = KBEmbedder(local_embedding_url=config.knowledge_base.embedding_local_url)
    retriever = KnowledgeRetriever(kb=kb, embedder=embedder, top_k=5)
    writer = KnowledgeWriter(kb=kb, embedder=embedder, writeback_threshold=0.15)
    return retriever, writer
```

### Step 8: 运行进化

```bash
python run_evolution.py --iterations 200
```

## 常见适配场景

### 场景1: 截面因子评估（factor_evolution 模式）

- 评估器: 继承 BaseEvaluator，实现截面 IC 计算
- 诊断: 使用默认 P0/P1/P2 规则
- 知识库: 量化因子种子知识
- 回测: 使用 VectorizedBacktester

### 场景2: 交易策略评估（quantitative_trading 模式）

- 评估器: 继承 BaseEvaluator，实现三阶段级联评估
- 诊断: 注册自定义规则（检查夏普率、最大回撤等）
- 评分: 自定义 ScoringConfig（夏普率 + 卡玛比率等）
- 回测: 使用 VectorizedBacktester（或事件驱动回测）

### 场景3: 无知识库模式

如果不需要知识库，只需：
1. 不复制 `knowledge_base/` 目录
2. 在 config.yaml 中设置 `knowledge_base.enabled: false`
3. 提示词中去掉 `{knowledge_base_context}` 变量

## 框架核心抽象

### DiagnosticsEngine
- 位置: `framework/diagnostics.py`
- 作用: P0/P1/P2 三级问题诊断 + LLM 友好格式化
- 扩展: `register_rule()` 注册自定义规则

### BaseEvaluator
- 位置: `framework/base_evaluator.py`
- 作用: 评估器基类，提供评分计算、指标统计、诊断集成
- 扩展: 继承并实现 `evaluate_program()`

### VectorizedBacktester
- 位置: `framework/backtest.py`
- 作用: 向量化截面因子轮动回测
- 使用: 直接实例化或调用 `vectorized_backtest()`

### PromptBuilder
- 位置: `framework/prompt_builder.py`
- 作用: 模板化提示词构建，支持变量填充和 KB 注入
- 使用: 加载模板文件或直接用字符串设置模板

### Config
- 位置: `framework/config_schema.py`
- 作用: 类型安全的配置加载和验证
- 使用: `config = Config.from_yaml("config.yaml")`

### KnowledgeBase (独立模块)
- 位置: `knowledge_base/`
- 组件:
  - `KnowledgeBase`: SQLite CRUD + 嵌入缓存管理
  - `KBEmbedder`: 嵌入向量生成（本地服务 > OpenAI > 哈希 fallback）
  - `KnowledgeRetriever`: 两阶段检索（规则过滤 + 向量排序）
  - `KnowledgeWriter`: 知识回写（改进检测 → 总结 → 存储）

## 关键设计决策

1. **知识库是可选模块**: 不复制 knowledge_base 目录即可禁用
2. **框架模块零依赖**: framework/ 只依赖 numpy/pandas/scipy 等基础库
3. **诊断系统领域无关**: P0/P1/P2 级别可适配任何优化问题
4. **模板系统不绑定具体 LLM**: 纯文本模板，任何模型可用
5. **回测工具独立可用**: 不依赖 OpenEvolve 框架，可单独用于策略回测
