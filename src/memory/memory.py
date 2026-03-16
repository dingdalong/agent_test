from typing import List, Dict, Any, Optional, Callable
import tiktoken
import logging

logger = logging.getLogger(__name__)

class ConversationBuffer:
    def __init__(
        self,
        max_rounds: int = 10,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        tokenizer: Optional[Callable[[str], int]] = None
    ):
        """
        :param max_rounds: 最多保留的对话轮数
        :param max_tokens: 最大 token 数（用于截断）
        :param system_prompt: 系统提示词（始终保留）
        :param tokenizer: 自定义 token 计数函数，输入字符串返回 token 数。若为 None，则使用 tiktoken 的 cl100k_base 近似。
        """
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, Any]] = []

        # 设置 token 计数器
        if tokenizer is None:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                self._token_counter = lambda text: len(enc.encode(text))  # 改名
            except Exception as e:
                logger.warning(f"Failed to load tiktoken, using approximate count: {e}")
                self._token_counter = lambda text: len(text) // 3 + 1
        else:
            self._token_counter = tokenizer

    def _count_messages_tokens(self, messages: List[Dict]) -> int:
        total = 0
        for msg in messages:
            if "content" in msg and msg["content"]:
                total += self._token_counter(msg["content"])  # 调用改名后的计数器
        return total

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, message: Dict[str, Any]):
        self.messages.append(message)

    def add_tool_message(self, tool_call_id: str, content: str):
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def get_messages_for_api(self) -> List[Dict[str, Any]]:
        """
        返回适合 API 的消息列表：
        - 始终包含 system_prompt（如果有）
        - 保留最近 max_rounds 轮对话，且保证工具调用链完整
        - 如果 token 超出 max_tokens，则从最旧的消息开始丢弃
        """
        # 从最新到最旧遍历消息，收集完整的轮次
        reversed_msgs = list(reversed(self.messages))
        selected = []
        rounds_count = 0
        i = 0
        while i < len(reversed_msgs) and rounds_count < self.max_rounds:
            msg = reversed_msgs[i]
            if msg["role"] == "assistant" and "tool_calls" in msg:
                # 找到所有连续的 tool 消息
                tool_msgs = []
                j = i + 1
                while j < len(reversed_msgs) and reversed_msgs[j]["role"] == "tool":
                    tool_msgs.append(reversed_msgs[j])
                    j += 1
                selected.append(msg)
                selected.extend(tool_msgs)
                i = j
            else:
                selected.append(msg)
                i += 1
            rounds_count += 1

        # 恢复原始顺序
        selected = list(reversed(selected))

        # 计算系统提示词的 token 数
        system_tokens = self._token_counter(self.system_prompt) if self.system_prompt else 0

        # 从最旧的消息开始丢弃，直到总 token 数 <= max_tokens
        while selected and self._count_messages_tokens(selected) + system_tokens > self.max_tokens:
            removed = selected.pop(0)
            logger.debug(f"Removed message due to token limit: {removed.get('role')} - {removed.get('content', '')[:50]}")

        # 组装最终结果
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        result.extend(selected)
        return result

    def clear(self):
        self.messages.clear()

import os
import uuid
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, Set

import chromadb
from chromadb.config import Settings

from .embeddings import OllamaEmbeddingFunction  # 假设已存在
from memory.memory_extractor import Fact, FactExtractor  # 导入我们新定义的数据类

logger = logging.getLogger(__name__)


# ---------- 辅助函数 ----------
def flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
    """
    将元数据中的复杂类型（列表、字典）转换为 JSON 字符串，
    确保所有值都是 Chroma 支持的基本类型。
    """
    flattened = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            flattened[k] = v
        else:
            try:
                flattened[k] = json.dumps(v, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to serialize metadata field '{k}': {e}, skipping")
    return flattened


# ---------- 向量记忆存储 ----------
class VectorMemory:
    """
    使用 ChromaDB 存储记忆向量，支持版本控制、活跃标记和检索。
    版本控制基于 speaker, type, attribute 的组合（base_id）。
    """

    def __init__(self,
                 collection_name: str = "user_memories",
                 persist_directory: str = "./chroma_data"):
        self.embedding_fn = OllamaEmbeddingFunction(
            os.getenv("OPENAI_MODEL_EMBEDDING"),
            os.getenv("OPENAI_MODEL_EMBEDDING_URL")
        )
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_conversation(self,
                user_input: str,
                assistant_response: str = "",
                source_id: Optional[str] = None,
                include_types: Optional[Set[str]] = None,
                enable_sensitive_filter: bool = True):

        extractor = FactExtractor()
        facts1 = extractor.extract(user_input, assistant_response, source_id, include_types, enable_sensitive_filter)
        for f in facts1:
            self._add_fact(f)  # attribute 应为 "user.name"，版本 1
    # ---------- 核心操作 ----------
    def _add_fact(self, fact: Fact) -> str:
        """
        添加一条事实记忆，自动处理版本控制和元数据展平。
        返回新插入的记忆 ID。
        """
        # 生成基础ID：基于 speaker, type, attribute（不含版本）
        base_id = self._generate_base_id(fact.speaker, fact.type, fact.attribute)

        # 获取现有活跃版本
        existing = self._get_active_versions(base_id)

        # 准备元数据
        new_meta = self._prepare_metadata(fact, base_id)

        # 冲突解决：决定是否插入新版本
        if existing:
            best_existing = self._select_best_existing(existing)
            if not self._should_replace(new_meta, best_existing):
                logger.debug(f"Existing fact (v{best_existing['metadata']['version']}) is better, skipping insertion for attribute '{fact.attribute}'")
                return best_existing["id"]  # 返回已存在的 ID
            else:
                # 将现有活跃版本全部标记为非活跃
                self._deactivate_versions([mem["id"] for mem in existing])
                new_meta["version"] = best_existing["metadata"]["version"] + 1
        else:
            new_meta["version"] = 1

        # 插入新记忆
        memory_id = str(uuid.uuid4())
        self.collection.add(
            documents=[fact.fact_text],
            metadatas=[new_meta],
            ids=[memory_id]
        )
        logger.info(f"Inserted new memory {memory_id} (base_id={base_id}, attribute={fact.attribute}, v{new_meta['version']})")
        return memory_id

    def search(self,
            query: str,
            n_results: int = 5,
            type_filter: Optional[str] = None,
            include_inactive: bool = False,
            detailed: bool = False) -> List[Dict[str, Any]]:
        """
        检索记忆，返回记忆列表。

        Args:
            query: 查询字符串
            n_results: 返回的最大结果数
            type_filter: 按记忆类型过滤（如 "user.preference"）
            include_inactive: 是否包含已标记为非活跃的记忆
            detailed: 若为 True，返回所有元数据；否则只返回核心字段

        Returns:
            列表，每个元素为字典：
            - 当 detailed=False 时包含 id, fact, type, confidence, speaker,
            attribute, timestamp, version
            - 当 detailed=True 时包含 id, fact, metadata（全部元数据）
        """
        where = {}
        if not include_inactive:
            where["is_active"] = True
        if type_filter:
            where["type"] = type_filter

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                fact_text = results["documents"][0][i]
                metadata = results["metadatas"][0][i]

                if detailed:
                    memories.append({
                        "id": doc_id,
                        "fact": fact_text,
                        "metadata": metadata
                    })
                else:
                    # 提取核心字段，过滤掉 None 值
                    core = {
                        "id": doc_id,
                        "fact": fact_text,
                        "type": metadata.get("type"),
                        "confidence": metadata.get("confidence"),
                        "speaker": metadata.get("speaker"),
                        "attribute": metadata.get("attribute"),
                        "timestamp": metadata.get("timestamp"),
                        "version": metadata.get("version"),
                    }
                    # 移除值为 None 的字段（可选，保持干净）
                    core = {k: v for k, v in core.items() if v is not None}
                    memories.append(core)

        return memories

    def get_by_id(self, memory_id: str) -> Optional[Dict]:
        result = self.collection.get(ids=[memory_id])
        if result["ids"]:
            return {
                "id": result["ids"][0],
                "fact": result["documents"][0],
                "metadata": result["metadatas"][0]
            }
        return None

    def get_history(self, base_fact_id: str) -> List[Dict]:
        """获取同一基础事实的所有版本历史（base_fact_id 由 speaker, type, attribute 生成）"""
        results = self.collection.get(where={"base_fact_id": base_fact_id})
        history = []
        for i, doc_id in enumerate(results["ids"]):
            history.append({
                "id": doc_id,
                "fact": results["documents"][i],
                "metadata": results["metadatas"][i]
            })
        history.sort(key=lambda x: x["metadata"]["version"])
        return history

    def delete(self, memory_id: str):
        self.collection.delete(ids=[memory_id])

    def deactivate(self, memory_id: str):
        """软删除：将记忆标记为非活跃"""
        self.collection.update(
            ids=[memory_id],
            metadatas=[{"is_active": False}]
        )

    def clear_all(self):
        self.collection.delete(where={})

    # ---------- 内部辅助方法 ----------
    def _generate_base_id(self, speaker: str, fact_type: str, attribute: str) -> str:
        """生成不包含版本的基础ID，基于 speaker, type, attribute"""
        base = f"{speaker}|{fact_type}|{attribute}"
        return hashlib.sha256(base.encode()).hexdigest()

    def _get_active_versions(self, base_id: str) -> List[Dict]:
        """获取指定基础ID的所有活跃记忆"""
        results = self.collection.get(
            where={"$and": [{"base_fact_id": base_id}, {"is_active": True}]}
        )
        memories = []
        for i, mem_id in enumerate(results["ids"]):
            memories.append({
                "id": mem_id,
                "metadata": results["metadatas"][i]
            })
        return memories

    def _prepare_metadata(self, fact: Fact, base_id: str) -> Dict[str, Any]:
        """将 Fact 对象转换为 Chroma 可存储的元数据字典（已展平）"""
        # 从 fact 中提取需要存储的字段
        meta = {
            "base_fact_id": base_id,
            "is_active": True,
            "timestamp": fact.timestamp,
            "type": fact.type,
            "speaker": fact.speaker,
            "attribute": fact.attribute,
            "confidence": fact.confidence,
            "source": fact.source,
            "original_utterance": fact.original_utterance,
            "is_plausible": fact.is_plausible,
            # 嵌套元数据展平
            "entities": json.dumps(fact.metadata.get("entities", [])),
            "negation": fact.metadata.get("negation", False),
            "temporal": fact.metadata.get("temporal", "present"),
            "extracted_fact_id": fact.fact_id,
        }
        # 移除 None 值
        meta = {k: v for k, v in meta.items() if v is not None}
        return flatten_metadata(meta)  # 确保所有值符合 Chroma 类型

    def _select_best_existing(self, memories: List[Dict]) -> Dict:
        """从一组活跃记忆中选出最佳版本（版本号最高）"""
        best = max(memories, key=lambda m: m["metadata"]["version"])
        return best

    def _should_replace(self, new_meta: Dict, best_existing: Dict) -> bool:
        """
        判断新事实是否应替换现有最佳版本。
        基于置信度和时间戳。
        """
        old_meta = best_existing["metadata"]
        new_conf = new_meta.get("confidence", 0.5)
        old_conf = old_meta.get("confidence", 0.5)
        new_time = new_meta.get("timestamp", "")
        old_time = old_meta.get("timestamp", "")

        if new_conf > old_conf:
            return True
        if new_conf == old_conf and new_time > old_time:
            return True
        return False

    def _deactivate_versions(self, memory_ids: List[str]):
        """将多个记忆标记为非活跃"""
        for mem_id in memory_ids:
            self.collection.update(
                ids=[mem_id],
                metadatas=[{"is_active": False}]
            )
