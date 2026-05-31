"""
知识库管理器

提供量化因子知识的结构化存储、检索和管理。
基于 SQLite 存储，支持向量嵌入和混合检索。
"""

import json
import os
import sqlite3
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """量化因子知识库"""

    def __init__(self, db_path: str, embedding_dim: int = 1536):
        """
        Args:
            db_path: SQLite 数据库文件路径
            embedding_dim: 嵌入向量维度
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim

        # 初始化数据库
        self._init_db()

        # 嵌入向量缓存文件路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._embedding_cache_path = os.path.join(script_dir, "kb_embeddings.npy")
        self._embedding_id_map_path = os.path.join(script_dir, "kb_embedding_ids.json")

        # 加载嵌入向量缓存
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_ids: List[str] = []
        self._load_embedding_cache()

    def _init_db(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        schema_path = os.path.join(os.path.dirname(__file__), "kb_schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                schema_sql = f.read()
            cursor.executescript(schema_sql)

        conn.commit()
        conn.close()

        # 启用 WAL 模式以支持多进程并发访问
        self._enable_wal()

        # 执行迁移（添加新列等）
        self._migrate_schema()

        logger.info(f"Knowledge base initialized: {self.db_path}")

    def _enable_wal(self):
        """启用 WAL 模式以支持更好的并发读写"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        conn.commit()
        conn.close()

    def _migrate_schema(self):
        """执行数据库迁移，安全地添加新列"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取 knowledge_analytics 表的现有列
        cursor.execute("PRAGMA table_info(knowledge_analytics)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        migrations = {
            "consecutive_failures": "ALTER TABLE knowledge_analytics ADD COLUMN consecutive_failures INTEGER DEFAULT 0",
            "last_retrieved_at": "ALTER TABLE knowledge_analytics ADD COLUMN last_retrieved_at TEXT DEFAULT ''",
            "last_improvement_at": "ALTER TABLE knowledge_analytics ADD COLUMN last_improvement_at TEXT DEFAULT ''",
        }

        for col_name, sql in migrations.items():
            if col_name not in existing_columns:
                try:
                    cursor.execute(sql)
                    logger.info(f"Migration: added column {col_name} to knowledge_analytics")
                except Exception as e:
                    logger.warning(f"Migration failed for {col_name}: {e}")

        conn.commit()
        conn.close()

    def _load_embedding_cache(self):
        """加载嵌入向量缓存"""
        try:
            if os.path.exists(self._embedding_cache_path) and os.path.exists(
                self._embedding_id_map_path
            ):
                self._embeddings = np.load(self._embedding_cache_path)
                with open(self._embedding_id_map_path, "r") as f:
                    self._embedding_ids = json.load(f)
                logger.debug(
                    f"Loaded {len(self._embedding_ids)} cached embeddings"
                )
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self._embeddings = None
            self._embedding_ids = []

    def _save_embedding_cache(self):
        """保存嵌入向量缓存"""
        try:
            if self._embeddings is not None and len(self._embedding_ids) > 0:
                np.save(self._embedding_cache_path, self._embeddings)
                with open(self._embedding_id_map_path, "w") as f:
                    json.dump(self._embedding_ids, f, ensure_ascii=False)
                logger.debug(f"Saved {len(self._embedding_ids)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    # ── CRUD 操作 ──

    def add_entry(self, entry: Dict[str, Any]) -> str:
        """
        添加一条知识

        Args:
            entry: 知识条目字典

        Returns:
            知识条目 ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 生成ID（如果没有提供）
        if "id" not in entry:
            content_hash = hashlib.md5(
                entry.get("search_text", "").encode()
            ).hexdigest()[:12]
            entry["id"] = f"kb_{content_hash}"

        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO knowledge_entries
            (id, problem_codes, factor_category, tags, market_condition,
             context_before, improvement_action, improvement_result, code_example,
             search_text, success_rating, source, source_url,
             usage_count, last_used_at, created_at, updated_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["id"],
                json.dumps(entry.get("problem_codes", []), ensure_ascii=False),
                entry.get("factor_category", "unknown"),
                json.dumps(entry.get("tags", []), ensure_ascii=False),
                entry.get("market_condition", "all"),
                entry.get("context_before", ""),
                entry.get("improvement_action", ""),
                entry.get("improvement_result", ""),
                entry.get("code_example", ""),
                entry.get("search_text", ""),
                entry.get("success_rating", 0.5),
                entry.get("source", "manual"),
                entry.get("source_url", ""),
                0,  # usage_count
                "",  # last_used_at
                entry.get("created_at", now),
                now,
                entry.get("status", "active"),
            ),
        )

        conn.commit()
        conn.close()

        logger.info(f"Added knowledge entry: {entry['id']}")
        return entry["id"]

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """获取单条知识"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM knowledge_entries WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return self._row_to_dict(row)

    def update_usage(self, entry_id: str):
        """更新知识使用次数"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        cursor.execute(
            """
            UPDATE knowledge_entries
            SET usage_count = usage_count + 1, last_used_at = ?
            WHERE id = ?
            """,
            (now, entry_id),
        )

        conn.commit()
        conn.close()

    def get_all_active_entries(self) -> List[Dict[str, Any]]:
        """获取所有活跃的知识条目"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM knowledge_entries WHERE status = 'active' ORDER BY success_rating DESC"
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def get_entries_by_ids(self, entry_ids: List[str]) -> List[Dict[str, Any]]:
        """根据ID列表批量获取条目"""
        if not entry_ids:
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        placeholders = ",".join("?" for _ in entry_ids)
        cursor.execute(
            f"SELECT * FROM knowledge_entries WHERE id IN ({placeholders})",
            entry_ids,
        )
        rows = cursor.fetchall()
        conn.close()

        # 保持输入顺序
        row_map = {row["id"]: self._row_to_dict(row) for row in rows}
        return [row_map[eid] for eid in entry_ids if eid in row_map]

    def search_by_tags(
        self,
        problem_codes: Optional[List[str]] = None,
        factor_category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        market_condition: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        通过规则/标签进行结构化检索（混合检索的第一步）

        Args:
            problem_codes: 问题类型码列表
            factor_category: 因子类别
            tags: 标签列表
            market_condition: 市场状态
            limit: 最大返回数

        Returns:
            匹配的知识条目列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        conditions = ["status = 'active'"]
        params: List[Any] = []

        # 问题码匹配（JSON数组包含任一匹配即可）
        if problem_codes:
            code_conditions = []
            for code in problem_codes:
                code_conditions.append("problem_codes LIKE ?")
                params.append(f"%{code}%")
            conditions.append(f"({' OR '.join(code_conditions)})")

        # 因子类别匹配
        if factor_category and factor_category != "unknown":
            conditions.append("(factor_category = ? OR factor_category = 'all')")
            params.append(factor_category)

        # 标签匹配
        if tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
            conditions.append(f"({' OR '.join(tag_conditions)})")

        # 市场状态匹配
        if market_condition:
            conditions.append("(market_condition = ? OR market_condition = 'all')")
            params.append(market_condition)

        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT * FROM knowledge_entries
            WHERE {where_clause}
            ORDER BY success_rating DESC
            LIMIT ?
        """
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def count(self) -> int:
        """获取知识库条目总数"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM knowledge_entries WHERE status = 'active'")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    # ── 嵌入向量管理 ──

    def update_embeddings(self, entry_id: str, embedding: np.ndarray):
        """更新单条知识的嵌入向量"""
        if entry_id in self._embedding_ids:
            idx = self._embedding_ids.index(entry_id)
            if self._embeddings is not None:
                self._embeddings[idx] = embedding
        else:
            if self._embeddings is None:
                self._embeddings = embedding.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, embedding.reshape(1, -1)])
            self._embedding_ids.append(entry_id)

    def get_embeddings_by_ids(
        self, entry_ids: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """根据ID列表获取对应的嵌入向量"""
        if self._embeddings is None or len(self._embedding_ids) == 0:
            return np.array([]), []

        indices = []
        valid_ids = []
        for eid in entry_ids:
            if eid in self._embedding_ids:
                indices.append(self._embedding_ids.index(eid))
                valid_ids.append(eid)

        if not indices:
            return np.array([]), []

        return self._embeddings[indices], valid_ids

    def get_quality_scores(self, entry_ids: List[str]) -> Dict[str, float]:
        """获取指定条目的 quality_score（用于检索排序加权）"""
        if not entry_ids:
            return {}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ",".join("?" for _ in entry_ids)
        cursor.execute(
            f"SELECT entry_id, quality_score FROM knowledge_analytics WHERE entry_id IN ({placeholders})",
            entry_ids,
        )
        scores = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        # 没有 analytics 记录的条目默认 0.5
        return {eid: scores.get(eid, 0.5) for eid in entry_ids}

    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """获取所有嵌入向量"""
        if self._embeddings is None:
            return np.array([]), []
        return self._embeddings, self._embedding_ids

    def sync_embeddings(self, entries: List[Dict[str, Any]]):
        """同步嵌入向量：移除无效ID，确保所有活跃条目都有嵌入"""
        active_ids = {e["id"] for e in entries if e.get("status") == "active"}

        # 保留活跃条目的嵌入
        if self._embeddings is not None:
            keep_indices = [
                i for i, eid in enumerate(self._embedding_ids) if eid in active_ids
            ]
            if keep_indices:
                self._embeddings = self._embeddings[keep_indices]
                self._embedding_ids = [self._embedding_ids[i] for i in keep_indices]
            else:
                self._embeddings = None
                self._embedding_ids = []

        self._save_embedding_cache()

    # ── 回写队列管理 ──

    def add_to_writeback_queue(self, entry: Dict[str, Any]) -> int:
        """添加待总结的改进经验到回写队列"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO writeback_queue
            (parent_code, child_code, parent_metrics, child_metrics,
             problem_codes, improvement_ratio, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["parent_code"],
                entry["child_code"],
                json.dumps(entry.get("parent_metrics", {}), ensure_ascii=False),
                json.dumps(entry.get("child_metrics", {}), ensure_ascii=False),
                json.dumps(entry.get("problem_codes", []), ensure_ascii=False),
                entry.get("improvement_ratio", 0.0),
                "pending",
            ),
        )

        queue_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return queue_id

    def get_pending_writebacks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取待处理的回写队列"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM writeback_queue WHERE status = 'pending' ORDER BY improvement_ratio DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            d = dict(row)
            d["parent_metrics"] = json.loads(d["parent_metrics"])
            d["child_metrics"] = json.loads(d["child_metrics"])
            d["problem_codes"] = json.loads(d["problem_codes"])
            results.append(d)

        return results

    def update_writeback_status(self, queue_id: int, status: str, knowledge_json: str = ""):
        """更新回写队列状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE writeback_queue
            SET status = ?, summarized_knowledge = ?
            WHERE id = ?
            """,
            (status, knowledge_json, queue_id),
        )

        conn.commit()
        conn.close()

    # ── 批量导入 ──

    def import_from_json(self, json_path: str) -> int:
        """从JSON文件批量导入知识"""
        if not os.path.exists(json_path):
            logger.error(f"Seed knowledge file not found: {json_path}")
            return 0

        with open(json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        count = 0
        for entry in entries:
            try:
                self.add_entry(entry)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to import entry {entry.get('id', 'unknown')}: {e}")

        logger.info(f"Imported {count}/{len(entries)} knowledge entries from {json_path}")
        return count

    # ── 知识修剪（Pruning）──

    def record_retrieval(self, entry_ids: List[str]):
        """
        记录知识被检索召回

        更新 knowledge_analytics 中的检索计数和最近检索时间。

        Args:
            entry_ids: 被检索召回的知识条目 ID 列表
        """
        if not entry_ids:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        for eid in entry_ids:
            # 确保 analytics 行存在
            cursor.execute(
                """
                INSERT OR IGNORE INTO knowledge_analytics (entry_id, total_retrievals, last_retrieved_at)
                VALUES (?, 0, ?)
                """,
                (eid, now),
            )
            cursor.execute(
                """
                UPDATE knowledge_analytics
                SET total_retrievals = total_retrievals + 1,
                    last_retrieved_at = ?
                WHERE entry_id = ?
                """,
                (now, eid),
            )

        conn.commit()
        conn.close()
        logger.debug(f"Recorded retrieval for {len(entry_ids)} entries")

    def record_feedback(self, entry_ids: List[str], improved: bool, improvement_ratio: float = 0.0):
        """
        记录知识召回后的改进反馈

        核心修剪逻辑：如果知识被召回但未带来改进，consecutive_failures +1；
        如果带来改进，重置 consecutive_failures = 0。

        Args:
            entry_ids: 被应用的知识条目 ID 列表
            improved: 子代因子是否有改进（combined_score 提升超过阈值）
            improvement_ratio: 改进幅度
        """
        if not entry_ids:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        for eid in entry_ids:
            # 确保 analytics 行存在
            cursor.execute(
                """
                INSERT OR IGNORE INTO knowledge_analytics (entry_id, total_retrievals, consecutive_failures)
                VALUES (?, 0, 0)
                """,
                (eid,),
            )

            if improved:
                # 有增益：成功率+1，连续失败清零，记录改进时间和幅度
                cursor.execute(
                    """
                    UPDATE knowledge_analytics
                    SET total_applications = total_applications + 1,
                        application_successes = application_successes + 1,
                        consecutive_failures = 0,
                        avg_improvement = (avg_improvement * application_successes + ?) / (application_successes + 1),
                        last_improvement_at = ?,
                        last_evaluated_at = ?,
                        quality_score = MIN(1.0, quality_score + 0.05)
                    WHERE entry_id = ?
                    """,
                    (improvement_ratio, now, now, eid),
                )
            else:
                # 无增益：应用次数+1，连续失败+1，质量评分降低
                cursor.execute(
                    """
                    UPDATE knowledge_analytics
                    SET total_applications = total_applications + 1,
                        consecutive_failures = consecutive_failures + 1,
                        last_evaluated_at = ?,
                        quality_score = MAX(0.0, quality_score - 0.05)
                    WHERE entry_id = ?
                    """,
                    (now, eid),
                )

        conn.commit()
        conn.close()

        status = "improved" if improved else "no_gain"
        logger.debug(
            f"Recorded feedback for {len(entry_ids)} entries: {status} "
            f"(ratio={improvement_ratio:.3f})"
        )

    def prune_stale_knowledge(
        self,
        min_retrievals: int = 3,
        max_consecutive_failures: int = 5,
        min_quality_score: float = 0.1,
        dry_run: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        自动修剪知识库中无效的知识条目

        修剪规则：
        1. 被召回 >= min_retrievals 次
        2. 连续无增益次数 >= max_consecutive_failures
        3. quality_score < 0.2（即几乎没有贡献）

        满足以上全部3个条件的条目将被标记为 deprecated 或删除。

        Args:
            min_retrievals: 最小召回次数阈值（被召回次数不足的不评估）
            max_consecutive_failures: 连续无增益次数上限
            min_quality_score: 最低质量评分（低于此值才会被修剪）
            dry_run: True=只返回候选列表不实际删除，False=执行修剪

        Returns:
            被修剪的条目列表（含修剪原因和统计数据）
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 查找满足修剪条件的条目
        cursor.execute(
            """
            SELECT ka.*, ke.search_text, ke.factor_category, ke.success_rating,
                   ke.usage_count, ke.status as entry_status
            FROM knowledge_analytics ka
            JOIN knowledge_entries ke ON ka.entry_id = ke.id
            WHERE ka.total_retrievals >= ?
              AND ka.consecutive_failures >= ?
              AND ka.quality_score <= ?
              AND ke.status = 'active'
              AND ke.source != 'seed'  -- 不删除种子知识
            ORDER BY ka.consecutive_failures DESC, ka.quality_score ASC
            """,
            (min_retrievals, max_consecutive_failures, min_quality_score),
        )

        candidates = [dict(row) for row in cursor.fetchall()]

        if not candidates:
            conn.close()
            logger.info("Prune check: no stale knowledge candidates found")
            return []

        if not dry_run:
            # 执行修剪：标记为 deprecated
            entry_ids = [c["entry_id"] for c in candidates]
            placeholders = ",".join("?" for _ in entry_ids)
            now = datetime.now().isoformat()

            cursor.execute(
                f"""
                UPDATE knowledge_entries
                SET status = 'deprecated', updated_at = ?
                WHERE id IN ({placeholders})
                """,
                [now] + entry_ids,
            )

            conn.commit()
            logger.info(
                f"Pruned {len(candidates)} stale knowledge entries: "
                f"{[c['entry_id'] for c in candidates]}"
            )
        else:
            logger.info(
                f"Prune dry-run: {len(candidates)} candidates would be pruned: "
                f"{[c['entry_id'] for c in candidates]}"
            )

        conn.close()
        return candidates

    def get_prune_stats(self) -> Dict[str, Any]:
        """获取知识修剪相关统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM knowledge_entries WHERE status = 'active'")
        active_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM knowledge_entries WHERE status = 'deprecated'")
        deprecated_count = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT AVG(consecutive_failures), MAX(consecutive_failures),
                   AVG(quality_score), MIN(quality_score)
            FROM knowledge_analytics
            WHERE total_retrievals > 0
            """
        )
        row = cursor.fetchone()
        avg_failures, max_failures, avg_quality, min_quality = row

        # 统计接近修剪阈值的条目数
        cursor.execute(
            """
            SELECT COUNT(*) FROM knowledge_analytics
            WHERE total_retrievals >= 3
              AND consecutive_failures >= 3
              AND quality_score < 0.3
            """
        )
        near_prune_count = cursor.fetchone()[0]

        conn.close()

        return {
            "active_entries": active_count,
            "deprecated_entries": deprecated_count,
            "avg_consecutive_failures": round(avg_failures or 0, 2),
            "max_consecutive_failures": max_failures or 0,
            "avg_quality_score": round(avg_quality or 0.5, 3),
            "min_quality_score": round(min_quality or 0.5, 3),
            "near_prune_threshold": near_prune_count,
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """将数据库行转换为字典"""
        d = dict(row)
        # 解析 JSON 字段
        for field in ["problem_codes", "tags"]:
            if field in d and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except json.JSONDecodeError:
                    d[field] = []
        return d

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        prune_stats = self.get_prune_stats()

        return {
            "total_entries": prune_stats["active_entries"],
            "deprecated_entries": prune_stats["deprecated_entries"],
            "category_distribution": self._get_category_distribution(),
            "average_success_rating": self._get_avg_rating(),
            "pending_writebacks": self._get_pending_writeback_count(),
            "embedding_cached": self._embeddings is not None and len(self._embedding_ids) > 0,
            "prune_stats": {
                "avg_consecutive_failures": prune_stats["avg_consecutive_failures"],
                "max_consecutive_failures": prune_stats["max_consecutive_failures"],
                "avg_quality_score": prune_stats["avg_quality_score"],
                "near_prune_threshold": prune_stats["near_prune_threshold"],
            },
        }

    def _get_category_distribution(self) -> Dict[str, int]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT factor_category, COUNT(*) FROM knowledge_entries WHERE status = 'active' GROUP BY factor_category"
        )
        result = dict(cursor.fetchall())
        conn.close()
        return result

    def _get_avg_rating(self) -> float:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT AVG(success_rating) FROM knowledge_entries WHERE status = 'active'"
        )
        avg = cursor.fetchone()[0] or 0
        conn.close()
        return round(avg, 3)

    def _get_pending_writeback_count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM writeback_queue WHERE status = 'pending'")
        count = cursor.fetchone()[0]
        conn.close()
        return count
