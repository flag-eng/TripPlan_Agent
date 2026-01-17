"""LLM服务模块"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from ..config import get_settings

# 全局LLM实例
_llm_instance = None


def get_llm_google() -> BaseChatModel:
    """
    获取LLM实例(单例模式)
    使用Google Gemini模型
    
    Returns:
        LangChain ChatModel实例
    """
    global _llm_instance
    
    if _llm_instance is None:
        settings = get_settings()
        
        # 从环境变量或配置中读取Gemini配置
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or settings.gemini_api_key
        model = os.getenv("GEMINI_MODEL") or settings.gemini_model
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY或GEMINI_API_KEY未配置,请在.env文件中设置")
        
        # 创建LangChain ChatGoogleGenerativeAI实例
        _llm_instance = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=0.7,
            convert_system_message_to_human=True  # Gemini需要将system消息转换为human消息
        )
        
        print(f"✅ LLM服务初始化成功")
        print(f"   提供商: Google Gemini")
        print(f"   模型: {model}")
    
    return _llm_instance


def get_llm_nvidia() -> BaseChatModel:
    """
    获取LLM实例(单例模式)
    使用NVIDIA API提供的DeepSeek R1模型（兼容OpenAI格式）

    Returns:
        LangChain ChatModel实例
    """
    global _llm_instance

    if _llm_instance is None:
        settings = get_settings()

        # 从环境变量或配置中读取NVIDIA配置
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVAPI_KEY") or settings.nvidia_api_key
        model = os.getenv("NVIDIA_MODEL") or settings.nvidia_model

        if not api_key:
            raise ValueError("NVIDIA_API_KEY或NVAPI_KEY未配置,请在.env文件中设置")

        # 创建LangChain ChatOpenAI实例（适配NVIDIA API）
        _llm_instance = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base="https://integrate.api.nvidia.com/v1",  # NVIDIA API端点
            model_name=model,
            temperature=0.7,
            # NVIDIA API无需转换system消息，保持默认即可
            # 可选：添加超时和重试配置，提升稳定性
            request_timeout=30,
            max_retries=3
        )

        print(f"✅ LLM服务初始化成功")
        print(f"   提供商: NVIDIA")
        print(f"   模型: {model}")
        print(f"   API端点: https://integrate.api.nvidia.com/v1")

    return _llm_instance

def reset_llm():
    """重置LLM实例(用于测试或重新配置)"""
    global _llm_instance
    _llm_instance = None

