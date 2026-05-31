-- 量化因子知识库 Schema
-- 用于存储因子改进经验、问题和解决方案的结构化知识

CREATE TABLE IF NOT EXISTS knowledge_entries (
    id TEXT PRIMARY KEY,

    -- 检索标签（规则匹配层）
    problem_codes TEXT NOT NULL DEFAULT '[]',        -- JSON array: ["IC_WEAK", "IR_LOW"]
    factor_category TEXT NOT NULL DEFAULT 'unknown',  -- value/momentum/volatility/quality/composite/technical
    tags TEXT NOT NULL DEFAULT '[]',                  -- JSON array: ["BP", "EP", "多因子组合"]
    market_condition TEXT DEFAULT 'all',              -- trending/range_bound/high_vol/all

    -- 知识内容（注入 prompt 的文本）
    context_before TEXT NOT NULL,       -- 问题场景描述
    improvement_action TEXT NOT NULL,   -- 改进方法和思路
    improvement_result TEXT NOT NULL,   -- 改进效果（量化描述）
    code_example TEXT DEFAULT '',       -- 代码示例

    -- 用于向量检索的拼接文本
    search_text TEXT NOT NULL,

    -- 元信息
    success_rating REAL DEFAULT 0.5,    -- 0-1，改进效果评级
    source TEXT DEFAULT 'manual',       -- evolution/manual/research/web
    source_url TEXT DEFAULT '',         -- 来源URL（如有）
    usage_count INTEGER DEFAULT 0,      -- 被检索使用的次数
    last_used_at TEXT DEFAULT '',       -- 最后被检索使用的时间
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),

    -- 知识状态
    status TEXT DEFAULT 'active'        -- active/deprecated/needs_review
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_factor_category ON knowledge_entries(factor_category);
CREATE INDEX IF NOT EXISTS idx_status ON knowledge_entries(status);
CREATE INDEX IF NOT EXISTS idx_success_rating ON knowledge_entries(success_rating DESC);
CREATE INDEX IF NOT EXISTS idx_usage_count ON knowledge_entries(usage_count DESC);

-- 知识去重和质量元数据表（后续离线沉淀使用）
CREATE TABLE IF NOT EXISTS knowledge_analytics (
    entry_id TEXT PRIMARY KEY,
    total_retrievals INTEGER DEFAULT 0,     -- 总检索次数
    total_applications INTEGER DEFAULT 0,   -- 被应用到prompt的次数
    application_successes INTEGER DEFAULT 0,-- 应用后带来改进的次数
    consecutive_failures INTEGER DEFAULT 0, -- 连续被召回但无增益的次数（核心修剪指标）
    avg_improvement REAL DEFAULT 0.0,       -- 平均改进幅度
    last_retrieved_at TEXT DEFAULT '',      -- 最近被检索时间
    last_improvement_at TEXT DEFAULT '',    -- 最近带来改进的时间
    last_evaluated_at TEXT DEFAULT '',
    quality_score REAL DEFAULT 0.5,         -- 综合质量评分
    needs_revision BOOLEAN DEFAULT 0,       -- 是否需要修正
    revision_notes TEXT DEFAULT '',         -- 修正备注
    FOREIGN KEY (entry_id) REFERENCES knowledge_entries(id)
);

-- 迁移：为 knowledge_analytics 添加修剪支持字段（如果表已存在但缺少这些列）
-- SQLite 不支持 IF NOT EXISTS for ALTER TABLE，迁移逻辑在 kb_manager._migrate_schema() 中处理

-- 知识回写队列（待LLM总结的候选知识）
CREATE TABLE IF NOT EXISTS writeback_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_code TEXT NOT NULL,
    child_code TEXT NOT NULL,
    parent_metrics TEXT NOT NULL,            -- JSON
    child_metrics TEXT NOT NULL,             -- JSON
    problem_codes TEXT NOT NULL DEFAULT '[]',-- JSON
    improvement_ratio REAL DEFAULT 0.0,
    status TEXT DEFAULT 'pending',           -- pending/summarized/stored/skipped
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    summarized_knowledge TEXT DEFAULT ''     -- LLM总结后的结构化知识JSON
);
