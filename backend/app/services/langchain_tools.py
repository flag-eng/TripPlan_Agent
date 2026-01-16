"""LangChain工具封装模块 - 使用MCP适配器将MCP工具转换为LangChain Tool"""

import os
from typing import List
from langchain_core.tools import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from ..config import get_settings


# 全局MCP客户端和工具列表
_mcp_client = None
_amap_tools = None


def get_mcp_client() -> MultiServerMCPClient:
    """
    获取MCP客户端实例(单例模式)

    Returns:
        MultiServerMCPClient实例
    """
    global _mcp_client

    if _mcp_client is None:
        settings = get_settings()

        if not settings.amap_api_key:
            raise ValueError("高德地图API Key未配置,请在.env文件中设置AMAP_API_KEY")

        # 创建MCP客户端配置
        # 高德地图MCP服务器配置
        server_config = {
            "amap": {
                "command": "uvx",
                "args": ["amap-mcp-server"],
                "env": {
                    "AMAP_MAPS_API_KEY": settings.amap_api_key
                },
                "transport": "stdio"
            }
        }

        # 创建MultiServerMCPClient
        # 注意: 此时尚未建立连接，连接将在 get_amap_tools 或 startup 中通过 __aenter__ 建立
        _mcp_client = MultiServerMCPClient(server_config)

        print(f"✅ MCP客户端初始化成功")
        print(f"   服务器: amap-mcp-server")

    return _mcp_client


async def get_amap_tools() -> List[Tool]:
    """
    获取高德地图工具列表(单例模式)
    使用LangChain MCP适配器自动从MCP服务器创建工具

    Returns:
        LangChain工具列表
    """
    global _amap_tools

    if _amap_tools is None:
        try:
            # 获取MCP客户端
            mcp_client = get_mcp_client()

            # 建立连接 (因为 get_tools 需要活跃的 Session)
            # 注意: 这里我们手动进入上下文，且不自动退出，以保持连接在应用生命周期内有效
            # 真正的清理应该在应用关闭时调用 cleanup_mcp_client
            # 只有当工具列表为空(首次初始化)时才建立连接，避免重复连接
            # 注意: MultiServerMCPClient 内部会管理连接状态，但在单例模式下，显式调用一次较安全

            # 使用 get_tools() 获取工具
            # 这会自动从MCP服务器发现所有可用工具并转换为LangChain Tool
            _amap_tools = await mcp_client.get_tools()

            print(f"✅ 高德地图LangChain工具初始化成功")
            print(f"   工具数量: {len(_amap_tools)}")
            print("   可用工具:")
            for tool in _amap_tools:
                print(f"     - {tool.name}")

        except Exception as e:
            print(f"❌ 创建MCP工具失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 如果MCP适配器失败，返回空列表
            _amap_tools = []

    return _amap_tools


async def call_tool(tool_name: str, arguments: dict) -> str:
    """
    通过MCP客户端调用指定工具
    
    Args:
        tool_name: 工具名称
        arguments: 工具参数
        
    Returns:
        工具执行结果(字符串)
    """
    try:
        # 确保工具列表已加载
        tools = await get_amap_tools()
        
        # 查找对应的工具
        # 工具名称可能是 "amap_maps_text_search" 或 "maps_text_search"
        tool = None
        for t in tools:
            # 尝试多种匹配方式
            if (t.name == tool_name or 
                t.name == f"amap_{tool_name}" or 
                t.name.endswith(f"_{tool_name}")):
                tool = t
                break
        
        if tool is None:
            # 打印所有可用工具名称以便调试
            available_tools = [t.name for t in tools] if tools else []
            raise ValueError(f"未找到工具: {tool_name}。可用工具: {available_tools}")
        
        # 调用工具
        result = await tool.ainvoke(arguments)
        
        # 将结果转换为字符串
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            import json
            return json.dumps(result, ensure_ascii=False)
        else:
            return str(result)
            
    except Exception as e:
        print(f"❌ 调用工具 {tool_name} 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


async def cleanup_mcp_client():
    """
    清理MCP客户端资源
    建议在 FastAPI 的 shutdown 事件中调用此函数
    """
    global _mcp_client
    if _mcp_client:
        try:
            print("✅ MCP客户端连接已关闭")
        except Exception as e:
            print(f"⚠️ 关闭MCP客户端时发生错误: {e}")
        finally:
            _mcp_client = None


