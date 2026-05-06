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
        logger.info(f"Knowledge base initialized: {self.db_path}")

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
            self._embedding_ids.append(entry_id)
            if self._embeddings is None:
                self._embeddings = embedding.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, embedding.reshape(1, -1)])

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

    # ── 辅助方法 ──

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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM knowledge_entries WHERE status = 'active'")
        active_count = cursor.fetchone()[0]

        cursor.execute(
            "SELECT factor_category, COUNT(*) FROM knowledge_entries WHERE status = 'active' GROUP BY factor_category"
        )
        category_dist = dict(cursor.fetchall())

        cursor.execute(
            "SELECT AVG(success_rating) FROM knowledge_entries WHERE status = 'active'"
        )
        avg_rating = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM writeback_queue WHERE status = 'pending'")
        pending_writebacks = cursor.fetchone()[0]

        conn.close()

        return {
            "total_entries": active_count,
            "category_distribution": category_dist,
            "average_success_rating": round(avg_rating, 3),
            "pending_writebacks": pending_writebacks,
            "embedding_cached": self._embeddings is not None and len(self._embedding_ids) > 0,
        }
