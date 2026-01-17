"""配置管理模块"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import List, Optional

# 加载环境变量
# 首先尝试加载当前目录的.env
load_dotenv()

# 可以加载其他.env文件(如果存在)
# 例如: helloagents_env = Path(__file__).parent.parent.parent.parent / "HelloAgents" / ".env"
# if helloagents_env.exists():
#     load_dotenv(helloagents_env, override=False)  # 不覆盖已有的环境变量


class Settings(BaseSettings):
    """应用配置"""

    # 应用基本配置
    app_name: str = "LangChain智能旅行助手"
    app_version: str = "1.0.0"
    debug: bool = False


    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    http_proxy: Optional[str] = "http://127.0.0.1:10808"
    https_proxy: Optional[str] = "http://127.0.0.1:10808"

    # CORS配置 - 使用字符串,在代码中分割
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"

    # 高德地图API配置
    amap_api_key: str = "026966ccdcdfaeea36ee884c134cdd85"

    # Unsplash API配置
    unsplash_access_key: str = "AI5mfD65XZTyqh-DC1KPqOws-gfiOh6U0DXiQLioTEc"
    unsplash_secret_key: str = "FOKgHsmz6yBGqMFxr5y_8YlDKMt7pnx4wJVpayDKGrM"

    # LLM配置 (Gemini)
    gemini_api_key: str = "xxxxxxxxxxxxxxxxxx"
    gemini_model: str = "gemma-3-27b-it"

    nvidia_api_key: str = "xxxxxxxxxxxxxxx"
    nvidia_model: str = "meta/llama-3.3-70b-instruct"

    # 日志配置
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # 忽略额外的环境变量

    def get_cors_origins_list(self) -> List[str]:
        """获取CORS origins列表"""
        return [origin.strip() for origin in self.cors_origins.split(',')]


# 创建全局配置实例
settings = Settings()

if settings.http_proxy:
    os.environ["http_proxy"] = settings.http_proxy
if settings.https_proxy:
    os.environ["https_proxy"] = settings.https_proxy


def get_settings() -> Settings:
    """获取配置实例"""
    return settings


# 验证必要的配置
def validate_config():
    """验证配置是否完整"""
    errors = []
    warnings = []

    if not settings.amap_api_key:
        errors.append("AMAP_API_KEY未配置")

    # Gemini API Key检查
    gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or settings.gemini_api_key
    if not gemini_api_key:
        warnings.append("GOOGLE_API_KEY或GEMINI_API_KEY未配置,LLM功能可能无法使用")

    if errors:
        error_msg = "配置错误:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    if warnings:
        print("\n⚠️  配置警告:")
        for w in warnings:
            print(f"  - {w}")

    return True


# 打印配置信息(用于调试)
def print_config():
    """打印当前配置(隐藏敏感信息)"""
    print(f"应用名称: {settings.app_name}")
    print(f"版本: {settings.app_version}")
    print(f"服务器: {settings.host}:{settings.port}")
    print(f"高德地图API Key: {'已配置' if settings.amap_api_key else '未配置'}")
    print(f"代理已设置: {os.environ.get('http_proxy')}")

    # 检查LLM配置
    gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or settings.gemini_api_key
    gemini_model = os.getenv("GEMINI_MODEL") or settings.gemini_model

    print(f"Gemini API Key: {'已配置' if gemini_api_key else '未配置'}")
    print(f"Gemini Model: {gemini_model}")
    print(f"日志级别: {settings.log_level}")

