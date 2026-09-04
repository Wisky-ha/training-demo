# 模型训练可视化平台后端

本目录是平台后端基础工程，使用 FastAPI 提供 API 入口，SQLAlchemy 连接 SQLite，并集中管理运行配置。

## 开发启动

```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # macOS/Linux 使用 cp
uvicorn app.main:app --reload
```

服务启动后可访问：

- `GET /api/health`：API 与数据库连通性检查
- `/docs`：FastAPI 自动生成的接口文档

## 配置

配置统一位于 `app/core/config.py`，环境变量使用 `MODEL_PLATFORM_` 前缀；可复制 `.env.example` 为 `.env` 进行本地覆盖。默认数据库为 `data/platform.db`，运行时目录（数据集、模型、脚本和上传文件）也由该配置统一创建。

## 基础自检

```bash
python -m unittest discover -s tests -v
```
