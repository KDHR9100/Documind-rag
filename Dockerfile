# Dockerfile
# 使用轻量级 Python 基础镜像
FROM python:3.11-slim-bookworm AS builder

# 设置工作目录
WORKDIR /app

# 【知识点1】: 安装 uv（极速 Python 包安装器）
# 官方安装脚本，比 pip 快 10-100 倍
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 【知识点2】: 复制依赖定义文件
COPY pyproject.toml uv.lock ./

# 【知识点3】: 利用 uv 在虚拟环境中安装依赖（挂载缓存加速）
# --frozen 表示严格遵循 uv.lock，不修改版本
RUN uv sync --no-dev

# 第二阶段：运行阶段（精简镜像）
FROM python:3.11-slim-bookworm

WORKDIR /app

# 从 builder 阶段复制虚拟环境
COPY --from=builder /app/.venv /app/.venv

# 【知识点4】: 将 .venv/bin 加入 PATH，让后续命令可直接使用
ENV PATH="/app/.venv/bin:$PATH"

# 复制项目源码
COPY app/ ./app/
COPY config/ ./config/
COPY docs/ ./docs/

# 创建向量库目录（如果不存在）
RUN mkdir -p /app/vector_store

# 暴露端口
EXPOSE 8000

# 【知识点5】: 启动 Uvicorn 服务
# --host 0.0.0.0 允许外部访问
# --workers 4 开启 4 个进程处理并发
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]