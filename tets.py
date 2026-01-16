import os
import google.generativeai as genai

# --- 记得填入你的代理和 Key ---
os.environ["http_proxy"] = "http://127.0.0.1:10808"
os.environ["https_proxy"] = "http://127.0.0.1:10808"
# --- 配置 API Key ---
# 你需要去 https://aistudio.google.com/app/apikey 申请免费 Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCbvnB9YlILeQdjnqIL37bEaOJW_uhXpfI" # 替换为你的 Gemini API Key

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("正在连接 Google 查询可用模型...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"发生错误: {e}")