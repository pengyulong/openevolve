#!/usr/bin/env python3
"""
知识库初始化脚本

功能：
1. 创建 SQLite 数据库
2. 导入种子知识（40条来自互联网的高质量因子改进经验）
3. 为所有知识条目生成嵌入向量
4. 显示知识库状态

用法：
    python init_kb.py                          # 使用 config.yaml 配置
    python init_kb.py --api-key sk-xxx         # 指定 API key
    python init_kb.py --no-embed               # 跳过嵌入生成
"""

import os
import sys
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Initialize knowledge base for factor evolution")
    parser.add_argument("--api-key", help="API key for embedding model")
    parser.add_argument("--base-url", help="API base URL")
    parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model name")
    parser.add_argument("--no-embed", action="store_true", help="Skip embedding generation")
    parser.add_argument("--seed-only", action="store_true", help="Only import seed data, skip embedding")
    args = parser.parse_args()

    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    parent_dir = os.path.dirname(script_dir)

    # 读取配置获取 API 配置
    import yaml
    config_path = os.path.join(parent_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    kb_config = config.get("knowledge_base", {})
    llm_config = config.get("llm", {})

    # 获取 API 配置
    api_key = args.api_key
    base_url = args.base_url

    if not api_key and llm_config.get("models"):
        api_key = llm_config["models"][0].get("api_key")
    if not base_url and llm_config.get("models"):
        base_url = llm_config["models"][0].get("api_base")

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    db_path = kb_config.get("db_path", "kb_store.db")
    seed_file = kb_config.get("seed_file", "seed_knowledge.json")
    embedding_model = args.model or kb_config.get("embedding_model", "text-embedding-3-small")

    # 解析为绝对路径
    if not os.path.isabs(db_path):
        db_path = os.path.join(parent_dir, db_path)
    if not os.path.isabs(seed_file):
        seed_file = os.path.join(parent_dir, seed_file)

    # Step 1: 创建知识库
    from kb_manager import KnowledgeBase

    logger.info(f"Creating knowledge base: {db_path}")
    kb = KnowledgeBase(db_path=db_path)

    # Step 2: 导入种子知识
    logger.info(f"Importing seed knowledge from: {seed_file}")
    count = kb.import_from_json(seed_file)
    logger.info(f"Imported {count} knowledge entries")

    # Step 3: 显示状态
    stats = kb.get_stats()
    logger.info(f"Knowledge base stats: {json.dumps(stats, ensure_ascii=False, indent=2)}")

    # Step 4: 生成嵌入向量
    if args.no_embed or args.seed_only:
        logger.info("Skipping embedding generation")
        logger.info("Note: Without embeddings, retrieval will fall back to rule-based filtering only")
        logger.info("To generate embeddings later, run: python init_kb.py")
    elif not api_key:
        logger.warning("No API key configured for embedding generation")
        logger.warning("Retrieval will use rule-based filtering (no semantic similarity)")
        logger.warning("To enable semantic search, run: python init_kb.py --api-key YOUR_KEY")
    else:
        from kb_embedder import KBEmbedder
        import numpy as np
        import time

        embedder = KBEmbedder(
            api_key=api_key,
            base_url=base_url,
            model=embedding_model,
            dim=kb_config.get("embedding_dim", 1536),
            local_embedding_url=kb_config.get("embedding_local_url"),
        )

        entries = kb.get_all_active_entries()
        if not entries:
            logger.warning("No entries to embed")
            return

        texts = [e.get("search_text", "") for e in entries]
        logger.info(f"Generating embeddings for {len(texts)} entries (model: {embedding_model})...")

        # 批量生成
        batch_size = 20
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                batch_embeddings = embedder.embed_texts(batch)
                all_embeddings.append(batch_embeddings)
                logger.info(f"  Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}: "
                           f"{len(batch)} embeddings generated")
            except Exception as e:
                logger.warning(f"  Batch {i//batch_size + 1} failed: {e}")
                # Fill with zeros for failed batch
                all_embeddings.append(np.zeros((len(batch), embedder.dim)))

        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
            entry_ids = [e["id"] for e in entries]

            # 更新嵌入缓存
            kb._embeddings = all_embeddings
            kb._embedding_ids = entry_ids
            kb._save_embedding_cache()

            logger.info(f"Saved {len(entry_ids)} embeddings to cache")
            logger.info(f"Embedding file: {kb._embedding_cache_path}")

    logger.info("\n=== Knowledge Base Initialized Successfully ===")
    logger.info(f"Total entries: {stats['total_entries']}")
    logger.info(f"Category distribution: {stats['category_distribution']}")
    logger.info(f"Database: {db_path}")


if __name__ == "__main__":
    main()
