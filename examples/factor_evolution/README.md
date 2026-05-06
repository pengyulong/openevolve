# 量化因子自动演化系统

基于 [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) 的二次开发，实现 A 股量化因子的自动化演化、知识召回与回写。

## 核心特性

- **多维度初始因子**：6 维度种子（价值/盈利/低波动/反转/小市值/低换手）
- **MAP-Elites 岛屿演化**：8 岛并行演化 + 迁移机制，维持种群多样性
- **知识库闭环**：两阶段知识检索（规则过滤 + 向量相似度）→ 注入 LLM Prompt → 改进经验自动回写
- **完整回测链路**：backtrader 框架，沪深 300 成分股，月度调仓
- **可视化仪表盘**：Streamlit 演化轨迹可视化（6 大模块，Plotly 交互图表）

## 环境依赖

- Python >= 3.10
- OpenEvolve（核心演化框架）
- Tushare（A 股数据源）

```bash
pip install openevolve tushare backtrader streamlit plotly openai
```

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 Tushare Token 和 LLM API Key
```

.env 文件内容：

```bash
TUSHARE_TOKEN=your_tushare_token    # Tushare 注册: https://tushare.pro
LLM_API_KEY=sk-your_api_key         # DeepSeek / OpenAI 兼容 API
LLM_API_BASE=https://api.deepseek.com/v1
```

加载环境变量：

```bash
source .env  # 或: export $(cat .env | xargs)
```

### 2. 初始化知识库

```bash
python knowledge_base/init_kb.py --config config.yaml
```

### 3. 运行演化

```bash
python run_evolution.py --config config.yaml --iterations 200
```

### 4. 查看演化进度

```bash
python report_progress.py
```

### 5. 评估最优因子

```bash
python evaluate_best_factor.py
```

### 6. 回测

```bash
python backtest_v2.py
```

### 7. 启动可视化仪表盘

```bash
streamlit run visualize_evolution.py
```

浏览器打开 `http://localhost:8501` 查看演化轨迹。

## 配置说明

`config.yaml` 中支持 `${ENV_VAR}` 语法引用环境变量，程序启动时自动替换。

主要配置项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `tushare_token` | Tushare API Token | `${TUSHARE_TOKEN}` |
| `llm.models[].api_key` | LLM API Key | `${LLM_API_KEY}` |
| `llm.models[].api_base` | LLM API 地址 | `${LLM_API_BASE}` |
| `knowledge_base.enabled` | 启用知识库 | `true` |
| `knowledge_base.writeback_threshold` | 回写阈值（Score 提升比例） | `0.15` |
| `knowledge_base.embedding_local_url` | 本地 Embedding 服务 | `${EMBEDDING_LOCAL_URL}` |
| `database.num_islands` | 并行岛屿数 | `8` |
| `database.population_size` | 种群大小 | `100` |

## 项目结构

```
factor_evolution/
├── config.yaml              # 演化配置
├── initial_factor.py         # 初始种子因子
├── best_program.py           # 演化最优因子
├── run_evolution.py          # 演化入口
├── data_loader.py            # 数据加载器（Tushare）
├── evaluator.py              # 因子评估器
├── evaluate_best_factor.py   # 最优因子评估
├── backtest_v2.py            # backtrader 回测
├── visualize_evolution.py    # Streamlit 可视化仪表盘
├── report_progress.py        # 进度报告
├── framework/                # 评估框架核心
│   ├── base_evaluator.py
│   ├── backtest.py
│   ├── config_schema.py      # 配置校验（含 ${ENV_VAR} 替换）
│   ├── diagnostics.py        # 因子诊断
│   └── prompt_builder.py
├── knowledge_base/           # 知识库系统
│   ├── kb_manager.py         # KB 管理器
│   ├── kb_retriever.py       # 两阶段检索
│   ├── kb_embedder.py        # 向量生成（本地/OpenAI/hash）
│   ├── kb_writer.py          # 知识回写器
│   ├── init_kb.py            # KB 初始化
│   ├── kb_schema.sql         # 数据库 Schema
│   └── seed_knowledge.json   # 种子知识
├── prompts/                  # LLM Prompt 模板
│   ├── system_message.txt
│   ├── diff_user.txt
│   └── full_rewrite_user.txt
├── .env.example              # 环境变量示例
└── .gitignore
```

## 知识库工作流程

```
演化循环
  ├── 父代因子评估 → 诊断问题码
  ├── 知识检索 ← KBRetriever
  │   ├── 规则过滤（问题码匹配 + 因子类别匹配）
  │   └── 向量相似度排序（Embedding）
  ├── 注入 LLM Prompt → 生成子代
  ├── 子代评估
  └── 知识回写 ← KnowledgeWriter（Score 提升 > 阈值时触发）
      ├── 提取改进经验
      ├── LLM 总结（或模板提取）
      ├── 生成 Embedding
      └── 写入知识库
```

## 依赖声明

本项目基于 [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) 框架进行二次开发和扩展：

- 演化核心引擎（Controller、Database、LLM 集成）复用 OpenEvolve
- 在 `openevolve/process_parallel.py` 中集成了知识库检索与回写流程
- 评估器、知识库、回测、可视化等均为独立开发模块
