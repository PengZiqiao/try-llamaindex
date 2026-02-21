import dotenv
import time
import asyncio
from typing import List
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig
from langfuse import get_client
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor


# 使用.env中的GOOGLE_API_KEY
dotenv.load_dotenv()

# 使用langfuse追踪
langfuse = get_client()
LlamaIndexInstrumentor().instrument()


class RateLimitedGoogleGenAI(GoogleGenAI):
    """
    带有速率限制的 GoogleGenAI LLM 包装类
    
    通过在每次 API 调用之间添加延迟来降低调用频率，避免超出 API 配额
    """
    
    def __init__(self, *args, request_delay: float = 1.0, **kwargs):
        """
        初始化带速率限制的 LLM 模型
        
        Args:
            request_delay: 每次 API 请求之间的延迟时间（秒）
            *args, **kwargs: 传递给 GoogleGenAI 的其他参数
        """
        super().__init__(*args, **kwargs)
        # 使用 object.__setattr__ 绕过 Pydantic 的字段验证，避免冲突
        object.__setattr__(self, '_request_delay', request_delay)
    
    def complete(self, prompt: str, **kwargs):
        """同步完成请求，添加延迟"""
        time.sleep(self._request_delay)
        return super().complete(prompt, **kwargs)
    
    def chat(self, messages, **kwargs):
        """
        同步聊天请求，添加延迟
        
        Args:
            messages: 聊天消息列表
            **kwargs: 其他参数
        
        Returns:
            LLM 生成的聊天响应
        """
        time.sleep(self._request_delay)
        return super().chat(messages, **kwargs)
    
    async def acomplete(self, prompt: str, **kwargs):
        """异步完成请求，添加延迟"""
        await asyncio.sleep(self._request_delay)
        return await super().acomplete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        """异步聊天请求，添加延迟"""
        await asyncio.sleep(self._request_delay)
        return await super().achat(messages, **kwargs)


class RateLimitedGoogleGenAIEmbedding(GoogleGenAIEmbedding):
    """
    带有速率限制的 GoogleGenAI Embedding 包装类
    
    通过在每次 API 调用之间添加延迟来降低调用频率，避免超出 API 配额
    支持自定义输出维度（如 768 维），可以减少存储空间和计算开销
    
    继承自 GoogleGenAIEmbedding，添加了 request_delay 和 output_dimensionality 参数
    """
    
    def __init__(self, *args, request_delay: float = 1.0, output_dimensionality: int = None, **kwargs):
        """
        初始化带速率限制的 Embedding 模型
        
        Args:
            request_delay: 每次 API 请求之间的延迟时间（秒）
            output_dimensionality: 输出维度（如 768），默认为 None（使用模型默认维度 3072）
            *args, **kwargs: 传递给 GoogleGenAIEmbedding 的其他参数
        """
        super().__init__(*args, **kwargs)
        # 使用 object.__setattr__ 绕过 Pydantic 的字段验证，避免冲突
        object.__setattr__(self, '_request_delay', request_delay)
        
        # 如果指定了输出维度，创建 EmbedContentConfig 配置对象
        if output_dimensionality is not None:
            object.__setattr__(self, '_embedding_config', EmbedContentConfig(
                output_dimensionality=output_dimensionality
            ))
        else:
            object.__setattr__(self, '_embedding_config', None)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的 embedding，添加延迟"""
        time.sleep(self._request_delay)
        
        if self._embedding_config is not None:
            result = self._client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=self._embedding_config
            )
            return result.embeddings[0].values
        
        return super()._get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取多个文本的 embeddings，批量处理时添加延迟"""
        time.sleep(self._request_delay)
        
        if self._embedding_config is not None:
            result = self._client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=self._embedding_config
            )
            return [embedding.values for embedding in result.embeddings]
        
        return super()._get_text_embeddings(texts)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询的 embedding，添加延迟"""
        time.sleep(self._request_delay)
        
        if self._embedding_config is not None:
            result = self._client.models.embed_content(
                model=self.model_name,
                contents=query,
                config=self._embedding_config
            )
            return result.embeddings[0].values
        
        return super()._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取单个文本的 embedding，添加延迟"""
        await asyncio.sleep(self._request_delay)
        
        if self._embedding_config is not None:
            result = await self._client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=self._embedding_config
            )
            return result.embeddings[0].values
        
        return await super()._aget_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步获取多个文本的 embeddings，添加延迟"""
        await asyncio.sleep(self._request_delay)
        
        if self._embedding_config is not None:
            result = await self._client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=self._embedding_config
            )
            return [embedding.values for embedding in result.embeddings]
        
        return await super()._aget_text_embeddings(texts)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询的 embedding，添加延迟"""
        await asyncio.sleep(self._request_delay)
        
        if self._embedding_config is not None:
            result = await self._client.models.embed_content(
                model=self.model_name,
                contents=query,
                config=self._embedding_config
            )
            return result.embeddings[0].values
        
        return await super()._aget_query_embedding(query)


# 定义llm和embed_model（使用 Google API，带速率限制）

# LLM 模型：使用 Google GenAI 的 gemini-3-flash-preview 模型
# 每次请求间隔 1 秒，避免超出 API 配额
llm = RateLimitedGoogleGenAI(
    "gemini-3-flash-preview",
    request_delay=1.0  # 每次请求间隔 1 秒
)

# Embedding 模型：使用 Google GenAI 的 gemini-embedding-001 模型
# 每次请求间隔 1 秒，避免超出 API 配额
# 输出维度设置为 768，减少存储空间和计算开销
embed_model = RateLimitedGoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    embed_batch_size=1,  # 批处理大小为 1，避免一次性处理过多请求
    request_delay=2.0,  # 每次请求间隔 2 秒
    output_dimensionality=768  # 设置输出维度为 768（默认为 3072）
)


if __name__ == "__main__":
    # 测试llm
    resp = llm.complete("直接回复我hello world")
    print(resp)

    # 测试embed_model的text_embedding
    embeddings = embed_model.get_text_embedding("这是文档")
    print(f"Embedding 维度: {len(embeddings)}")
    print(embeddings[:5])

    # 测试embed_model的query_embedding
    embeddings = embed_model.get_query_embedding("这是问题")
    print(f"Query Embedding 维度: {len(embeddings)}")
    print(embeddings[:5])