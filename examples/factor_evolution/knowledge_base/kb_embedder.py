"""
嵌入向量生成器

支持多种后端：
1. 本地 embedding server（优先）：HTTP POST /llm_service/embedding
2. OpenAI 兼容接口：/v1/embeddings
3. 哈希 fallback：当以上都不可用时
"""

import logging
import time
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class KBEmbedder:
    """知识库嵌入向量生成器"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        batch_size: int = 20,
        local_embedding_url: Optional[str] = None,
    ):
        """
        Args:
            api_key: API 密钥（OpenAI 兼容接口）
            base_url: API 基础 URL（OpenAI 兼容接口）
            model: 嵌入模型名称
            dim: 向量维度（使用本地服务时会自动检测并更新）
            batch_size: 批量处理大小
            local_embedding_url: 本地 embedding server 地址，
                如 "http://localhost:8190/llm_service/embedding"
                设置后将优先使用本地服务
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.local_embedding_url = local_embedding_url
        self._local_dim_detected = False  # 是否已从本地服务检测到维度

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        为文本列表生成嵌入向量

        优先级：本地 embedding server > OpenAI 兼容 API > 哈希 fallback

        Args:
            texts: 文本列表

        Returns:
            (N, dim) 的 numpy 数组
        """
        if not texts:
            return np.array([])

        # 优先使用本地 embedding server
        if self.local_embedding_url:
            try:
                return self._local_embed(texts)
            except Exception as e:
                logger.warning(
                    f"Local embedding server failed: {e}, "
                    f"falling back to OpenAI API / hash"
                )

        # 使用 OpenAI SDK
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                response = client.embeddings.create(model=self.model, input=batch)

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                if i + self.batch_size < len(texts):
                    time.sleep(0.2)  # 避免速率限制

            return np.array(all_embeddings, dtype=np.float32)

        except ImportError:
            logger.warning("openai package not installed, using hashing-based fallback")
            return self._hash_embed(texts)
        except Exception as e:
            logger.warning(f"Embedding API failed: {e}, using hashing-based fallback")
            return self._hash_embed(texts)

    def _local_embed(self, texts: List[str]) -> np.ndarray:
        """
        通过本地 embedding server 生成嵌入向量

        请求格式:
            POST {local_embedding_url}
            {"text": ["文本1", "文本2", ...]}

        响应格式:
            {"code": 0, "message": "SUCCESS", "result": [[...], [...]]}

        Args:
            texts: 文本列表

        Returns:
            (N, dim) 的 numpy 数组
        """
        import requests as _requests

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            response = _requests.post(
                self.local_embedding_url,
                json={"text": batch},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            if data.get("code") != 0:
                raise RuntimeError(
                    f"Local embedding server error: code={data.get('code')}, "
                    f"message={data.get('message')}"
                )

            batch_embeddings = data["result"]

            # 自动检测并更新维度
            if not self._local_dim_detected and batch_embeddings:
                detected_dim = len(batch_embeddings[0])
                if detected_dim != self.dim:
                    logger.info(
                        f"Detected embedding dimension {detected_dim} from local "
                        f"server (was {self.dim}), updating self.dim"
                    )
                    self.dim = detected_dim
                self._local_dim_detected = True

            all_embeddings.extend(batch_embeddings)

            if i + self.batch_size < len(texts):
                time.sleep(0.05)  # 本地服务间隔较短

        return np.array(all_embeddings, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """为单个文本生成嵌入向量"""
        result = self.embed_texts([text])
        return result[0] if len(result) > 0 else np.zeros(self.dim)

    def _hash_embed(self, texts: List[str]) -> np.ndarray:
        """
        基于哈希的简单嵌入（当 API 不可用时的 fallback）
        使用字符 n-gram 统计生成伪嵌入向量
        """
        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)

        for i, text in enumerate(texts):
            import hashlib

            # 使用 n-gram 哈希生成伪嵌入
            vec = np.zeros(self.dim, dtype=np.float32)
            for n in [2, 3, 4]:
                for j in range(len(text) - n + 1):
                    ngram = text[j : j + n]
                    idx = int(hashlib.md5(ngram.encode()).hexdigest(), 16) % self.dim
                    vec[idx] += 1.0
            # 归一化
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings[i] = vec

        return embeddings


import os  # noqa: E402 (used in __init__)
