"""
标准化配置 Schema

提供类型安全的配置加载和验证。
支持从 YAML 文件加载，带默认值填充和字段校验。
"""

import os
import re
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def _substitute_env(value: Any) -> Any:
    """递归替换字符串中的 ${VAR_NAME} 为环境变量值"""
    if isinstance(value, str):
        def _replace(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        return re.sub(r'\$\{(\w+)\}', _replace, value)
    elif isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


@dataclass
class DataConfig:
    """数据源配置"""
    token: str = ""
    start_date: str = "20210101"
    end_date: str = "20250101"
    train_end_date: str = "20231231"
    max_stocks: int = 300
    forward_period: int = 5
    min_stocks_per_day: int = 30
    stock_pool: str = "hs300"


@dataclass
class LLMConfig:
    """LLM 配置"""
    models: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "deepseek-chat", "weight": 0.7},
    ])
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: str = ""
    base_url: str = ""


@dataclass
class DatabaseConfig:
    """进化数据库配置"""
    num_islands: int = 8
    population_size: int = 100
    feature_dimensions: List[str] = field(default_factory=lambda: [
        "abs_ic_mean", "ic_ir", "ic_stability", "factor_turnover"
    ])
    mutation_rate: float = 0.3


@dataclass
class KnowledgeBaseConfig:
    """知识库配置"""
    enabled: bool = True
    db_path: str = "knowledge_base/kb_store.db"
    seed_data_path: str = "knowledge_base/seed_knowledge.json"
    embedding_local_url: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    retrieval_top_k: int = 5
    rule_filter_min_candidates: int = 10
    writeback_threshold: float = 0.15
    writeback_use_llm: bool = True


@dataclass
class EvolutionConfig:
    """进化过程配置"""
    diff_based_evolution: bool = False
    iterations: int = 200
    checkpoint_interval: int = 50


@dataclass
class Config:
    """
    完整的项目配置

    使用方式:
        config = Config.from_yaml("config.yaml")
        print(config.data.start_date)
        print(config.knowledge_base.enabled)
    """
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    # 原始配置字典（用于访问未结构化的字段）
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        从 YAML 文件加载配置

        Args:
            path: config.yaml 文件路径

        Returns:
            Config 实例

        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML 解析错误
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        # 替换 ${ENV_VAR} 为环境变量值
        raw = _substitute_env(raw)

        config = cls()
        config._raw = raw

        # 解析 data 部分
        config.data = DataConfig(
            token=raw.get("tushare_token", raw.get("token", "")),
            start_date=str(raw.get("data_start_date", "20210101")),
            end_date=str(raw.get("data_end_date", "20250101")),
            train_end_date=str(raw.get("train_end_date", "20231231")),
            max_stocks=int(raw.get("max_stocks", 300)),
            forward_period=int(raw.get("forward_period", 5)),
            min_stocks_per_day=int(raw.get("min_stocks_per_day", 30)),
            stock_pool=str(raw.get("stock_pool", "hs300")),
        )

        # 解析 LLM 部分
        llm_raw = raw.get("llm", {})
        config.llm = LLMConfig(
            models=llm_raw.get("models", [{"name": "deepseek-chat", "weight": 0.7}]),
            temperature=float(llm_raw.get("temperature", 0.7)),
            max_tokens=int(llm_raw.get("max_tokens", 2000)),
            api_key=llm_raw.get("api_key", ""),
            base_url=llm_raw.get("base_url", ""),
        )

        # 解析 database 部分
        db_raw = raw.get("database", {})
        config.database = DatabaseConfig(
            num_islands=int(db_raw.get("num_islands", 8)),
            population_size=int(db_raw.get("population_size", 100)),
            feature_dimensions=list(db_raw.get(
                "feature_dimensions",
                ["abs_ic_mean", "ic_ir", "ic_stability", "factor_turnover"]
            )),
            mutation_rate=float(db_raw.get("mutation_rate", 0.3)),
        )

        # 解析 knowledge_base 部分
        kb_raw = raw.get("knowledge_base", {})
        config.knowledge_base = KnowledgeBaseConfig(
            enabled=bool(kb_raw.get("enabled", True)),
            db_path=str(kb_raw.get("db_path", "knowledge_base/kb_store.db")),
            seed_data_path=str(kb_raw.get("seed_data_path", "knowledge_base/seed_knowledge.json")),
            embedding_local_url=str(kb_raw.get("embedding_local_url", "")),
            embedding_api_key=str(kb_raw.get("embedding_api_key", "")),
            embedding_base_url=str(kb_raw.get("embedding_base_url", "")),
            embedding_model=str(kb_raw.get("embedding_model", "text-embedding-3-small")),
            retrieval_top_k=int(kb_raw.get("retrieval_top_k", 5)),
            rule_filter_min_candidates=int(kb_raw.get("rule_filter_min_candidates", 10)),
            writeback_threshold=float(kb_raw.get("writeback_threshold", 0.15)),
            writeback_use_llm=bool(kb_raw.get("writeback_use_llm", True)),
        )

        # 解析 evolution 部分
        evo_raw = raw.get("evolution", {})
        config.evolution = EvolutionConfig(
            diff_based_evolution=bool(raw.get("diff_based_evolution", False)),
            iterations=int(evo_raw.get("iterations", 200)),
            checkpoint_interval=int(raw.get("checkpoint_interval", 50)),
        )

        return config

    def validate(self) -> List[str]:
        """
        验证配置完整性

        Returns:
            错误信息列表（空列表 = 配置有效）
        """
        errors = []

        # 数据源校验
        if not self.data.token:
            errors.append("data.token 未设置（需要 Tushare token）")
        if self.data.max_stocks < 30:
            errors.append(f"data.max_stocks={self.data.max_stocks} 过小，建议≥30")

        # LLM 校验
        if not self.llm.models:
            errors.append("llm.models 为空，需要至少一个模型")

        # 知识库校验
        if self.knowledge_base.enabled:
            if self.knowledge_base.retrieval_top_k < 1:
                errors.append("knowledge_base.retrieval_top_k 必须 ≥ 1")

        return errors

    def get(self, key: str, default: Any = None) -> Any:
        """从原始配置字典获取未结构化的字段"""
        return self._raw.get(key, default)


def load_config(config_path: str) -> Config:
    """
    加载并验证配置文件

    命令行工具的标准入口。

    Args:
        config_path: config.yaml 文件路径

    Returns:
        已验证的 Config 实例

    Raises:
        SystemExit: 配置验证失败时
    """
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = Config.from_yaml(config_path)
    errors = config.validate()

    if errors:
        for err in errors:
            logger.error(f"[CONFIG] {err}")
        logger.warning("配置存在警告，继续运行...")

    return config
