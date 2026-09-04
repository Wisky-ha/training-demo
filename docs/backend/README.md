# 模型训练可视化平台后端

步骤 1 基础工程提供 FastAPI 入口、统一配置、SQLite 连接，以及后续训练流程所需的科学计算依赖。

## 本地运行

在仓库根目录执行：

```bash
python -m pip install -r backend/requirements-dev.txt
uvicorn backend.app.main:app --reload
```

也可以使用项目配置安装开发依赖：

```bash
python -m pip install -e ".[dev]"
```

也可以在 `backend` 目录中执行 `uvicorn app.main:app --reload`。

默认配置使用 `data/app.db`，模型和上传文件分别保存到 `data/models`、`data/uploads`，目录会在应用启动时自动创建。复制 `backend/.env.example` 为项目根目录或 `backend/.env` 后即可通过环境变量覆盖配置；运行目录下的 `.env` 也支持作为本地覆盖。

## 基础自检

```bash
python -c "from backend.app.main import app; from backend.app.core.config import get_settings; from backend.app.db.connection import connect_database; import pandas, numpy, sklearn, joblib; s=get_settings(); c=connect_database(s); print(app.title, s.database_path, c.execute('SELECT 1').fetchone()[0]); c.close()"
pytest -q backend/tests
```
