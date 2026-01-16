import os
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

os.environ["http_proxy"] = "http://127.0.0.1:10808"
os.environ["https_proxy"] = "http://127.0.0.1:10808"
# --- 配置 API Key ---
# 你需要去 https://aistudio.google.com/app/apikey 申请免费 Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCbvnB9YlILeQdjnqIL37bEaOJW_uhXpfI" # 替换为你的 Gemini API Key

# 1. 初始化模型 (使用 Gemini 1.5 Flash，速度快且免费额度高)
# convert_system_message_to_human=True 是为了兼容某些不直接支持 System Message 的 Gemini 版本旧习惯，
# 但新版 LangChain 通常已自动处理。这里保持默认配置即可。

@tool
def get_current_weather(city: str) -> str:
    """查询指定城市的当前天气情况。"""
    # 这里是模拟数据，实际使用时可以调用 OpenWeatherMap 等 API
    print(f"\n[工具日志] 正在查询 {city} 的天气...")
    if "北京" in city:
        return f"{city} 今天晴朗，气温 25 度，风力 3 级。"
    elif "上海" in city:
        return f"{city} 今天多云，气温 22 度，有小雨。"
    else:
        return f"{city} 天气数据未知，建议带伞。"

@tool
def magic_calculator(expression: str) -> str:
    """用于处理复杂的数学计算。输入应该是一个数学表达式字符串。"""
    # 简单的 eval 实现，生产环境需注意安全
    print(f"\n[工具日志] 正在计算: {expression}")
    try:
        return str(eval(expression))
    except:
        return "计算出错"

# 将工具放入列表
tools = [get_current_weather, magic_calculator]

# --- 3. 初始化 Gemini 模型 ---
# 注意：目前公开的最新预览版通常是 gemini-2.0-flash-exp
# 如果你有 gemini-3-flash-preview 的权限，直接替换 model 参数即可
llm = (ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
))

# --- 4. 构建 Prompt ---
# 这是一个标准的 Agent Prompt，包含 system message 和 placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的 AI 助手。你可以使用工具来回答用户的问题。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"), # 关键：用于存放 Agent 的思考过程和工具调用结果
])

# --- 5. 创建 Agent ---
# create_tool_calling_agent 是 LangChain 专门为支持 Function Calling 的模型设计的
agent = create_tool_calling_agent(llm, tools, prompt)

# --- 6. 创建执行器 (Executor) ---
# AgentExecutor 负责运行 Agent 的循环（思考 -> 执行 -> 观察 -> 再思考）
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True # 开启 verbose 可以看到详细的思考过程
)

# --- 7. 运行测试 ---

print("--------------------------------------------------")
print("测试 1: 不需要工具的普通对话")
response1 = agent_executor.invoke({"input": "你好，请做一个简短的自我介绍。"})
print(f"AI 回复: {response1['output']}")

print("\n--------------------------------------------------")
print("测试 2: 需要调用工具 (天气)")
response2 = agent_executor.invoke({"input": "北京今天天气怎么样？适合穿什么？"})
print(f"AI 回复: {response2['output']}")

print("\n--------------------------------------------------")
print("测试 3: 复杂逻辑 (多步思考)")
# 这个问题通常需要 AI 先查天气，结合上下文或者自行判断，甚至可能调用计算（如果涉及数字）
response3 = agent_executor.invoke({"input": "如果上海现在的温度乘以 10，结果是多少？"})
print(f"AI 回复: {response3['output']}")