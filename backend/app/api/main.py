"""FastAPI主应用"""
import sys
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..config import get_settings, validate_config, print_config
from .routes import trip, poi, map as map_routes


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")

# 2. 配置 logging (每天轮转一次日志，保留7天)
logger = logging.getLogger("AppLogger")
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = TimedRotatingFileHandler(
    log_file, when="midnight", interval=1, backupCount=7, encoding='utf-8'
)
# 设置日志格式 (时间 - 级别 - 内容)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 3. 定义双向输出类 (同时写控制台和文件)
class LoggerWriter:
    def __init__(self, level, original_stream):
        self.level = level
        self.terminal = original_stream  # 保存原本的输出流(控制台)

    def write(self, message):
        # 步骤A: 输出到控制台 (保持原有 print 的效果)
        self.terminal.write(message)
        self.terminal.flush() # 确保控制台实时显示

        # 步骤B: 写入日志文件 (过滤掉纯换行符，避免日志空行过多)
        if message and message.strip():
            logger.log(self.level, message.strip())

    def flush(self):
        # 必须实现 flush 方法，因为 Python 的 IO流 需要它
        self.terminal.flush()

# 4. 执行重定向 (核心步骤)
# 将 print() 的默认去向(stdout) 劫持到我们的 LoggerWriter
sys.stdout = LoggerWriter(logging.INFO, sys.stdout)
# 将 报错信息(stderr) 劫持到 LoggerWriter (标记为 ERROR)
sys.stderr = LoggerWriter(logging.ERROR, sys.stderr)


# 获取配置
settings = get_settings()

# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于LangChain框架的智能旅行规划助手API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(trip.router, prefix="/api")
app.include_router(poi.router, prefix="/api")
app.include_router(map_routes.router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    print("\n" + "="*60)
    print(f"🚀 {settings.app_name} v{settings.app_version}")
    print("="*60)
    
    # 打印配置信息
    print_config()
    
    # 验证配置
    try:
        validate_config()
        print("\n✅ 配置验证通过")
    except ValueError as e:
        print(f"\n❌ 配置验证失败:\n{e}")
        print("\n请检查.env文件并确保所有必要的配置项都已设置")
        raise
    
    print("\n" + "="*60)
    print("📚 API文档: http://localhost:8000/docs")
    print("📖 ReDoc文档: http://localhost:8000/redoc")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    print("\n" + "="*60)
    print("👋 应用正在关闭...")
    print("="*60 + "\n")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )

