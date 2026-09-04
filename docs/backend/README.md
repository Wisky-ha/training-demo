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

## 脚本库接口（步骤 4）

- `POST /api/scripts/upload` 使用 multipart 上传 `file` 和 `name`、`version`、`script_type`、`supported_model_types` 元数据。仅接受 UTF-8、语法有效的 `.py` 文件；新记录默认启用。
- `GET /api/scripts` 返回分页的 `items` 和 `pagination`，项目包含名称、版本、脚本类型、适用模型类型、`created_at`/`uploaded_at`、启用状态和源码。默认只列启用脚本；传入 `model_type` 时只列声明兼容该模型的脚本。显式传 `status=disabled` 可供库管理查看停用脚本。
- 源码使用 UUID 文件名保存到 `APP_SCRIPT_STORAGE_DIR`；未配置时使用 `APP_STORAGE_ROOT`，再回退到模型存储根下的 `script/` 目录。文件名不参与路径拼接，数据库事务失败会清理已写入文件。
- 同一名称、脚本类型和版本不可重复，非法元数据/文件返回 422/400，重复版本返回 409。数据集、预处理执行和训练接口不在本步骤实现。
