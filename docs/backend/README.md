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

## 脚本库接口（步骤 7）

- `POST /api/scripts/upload` 使用 multipart 上传 `file` 和 `name`、`script_type`、`supported_model_types` 元数据；`version` 可选，省略时按同一名称/类型生成递增的 `v1`、`v2`……。仅接受 UTF-8、语法有效的 `.py` 文件；新记录默认启用。
- `GET /api/scripts` 返回分页的 `items` 和 `pagination`，项目包含名称、版本、脚本类型、适用模型类型、`created_at`/`uploaded_at`、启用状态和源码。默认只列启用脚本；传入 `model_type` 时只列声明兼容该模型的脚本。显式传 `status=disabled` 可供库管理查看停用脚本。
- `GET /api/scripts/{id}` 查询单个不可变脚本版本；`POST /api/scripts/{id}/enable` 和 `/disable` 管理可用状态，不删除源码或历史版本。
- 源码使用 UUID 文件名保存到 `APP_SCRIPT_STORAGE_DIR`；未配置时使用 `APP_STORAGE_ROOT`，再回退到模型存储根下的 `script/` 目录。文件名不参与路径拼接，数据库事务失败会清理已写入文件。
- 同一名称、脚本类型和版本不可重复，非法元数据/文件返回结构化 422/400，重复版本返回 409。

## CSV 数据集接口（步骤 5）

- `POST /api/datasets/upload` 接收 multipart 字段 `file`，仅接受 `.csv`；成功返回并保存数据集 ID、文件名、完整表头、按位置推断的时间/特征/目标角色、行列数、最多 5 行样例、时间解析与范围、字段类型、数值列、缺失值统计、校验结果和文件存储校验信息。
- 第一列必须全部可解析为时间，最后一列必须是有限数值；特征允许部分缺失，但不能整列为空。至少需要两行数据以保证后续 80%/20% 划分的训练集和测试集均非空。时间重复、目标缺失、编码/CSV 格式错误会返回结构化的 `errors`（同时保留可直接展示的 `detail`）。
- 数据文件按 UUID 保存为 `<storage-root>/dataset/<dataset-id>.csv`，数据库保存 `DatasetORM` 检查元数据及 `FileArtifactORM` 的大小和 SHA-256；数据库失败时会清理已写入文件。`APP_MAX_DATASET_SIZE_BYTES` 默认 50 MiB，HTTP 层最多读取限制值加 1 字节，预览固定最多 5 行，避免大文件响应和内存无界增长。
- 本步骤不实现预处理、数据划分执行、训练、评估、发布或预测接口。
