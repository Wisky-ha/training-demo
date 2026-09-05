"""Small administrative commands for local backend initialization.

The CLI deliberately performs only additive database work: SQLAlchemy creates
missing tables, the compatibility upgrader adds known nullable/defaulted
SQLite columns, and the baseline service inserts missing system rows.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sqlalchemy import func, select

from .core.config import Settings, get_settings
from .db.models import ModelTypeORM, ModelVersionORM
from .db.session import create_database_engine, create_session_factory, initialize_database
from .services.model_baseline import BASELINE_VERSION, ModelBaselineService


def initialize_local_database(settings: Settings | None = None) -> dict[str, int]:
    """Create/update the schema and idempotently provision model baselines."""

    active_settings = settings or get_settings()
    engine = create_database_engine(active_settings)
    try:
        initialize_database(engine=engine, settings=active_settings)
        factory = create_session_factory(engine)
        with factory() as session:
            baselines = ModelBaselineService(
                session, settings=active_settings
            ).initialize_baselines()
            return {
                "model_types": session.scalar(select(func.count(ModelTypeORM.id))) or 0,
                "baselines": session.scalar(
                    select(func.count(ModelVersionORM.id)).where(
                        ModelVersionORM.version == BASELINE_VERSION,
                        ModelVersionORM.is_baseline.is_(True),
                    )
                )
                or 0,
                "created_or_existing_baselines": len(baselines),
            }
    finally:
        engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="模型训练平台后端管理命令")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init-db", help="创建/升级 SQLite 表并初始化三种模型及 v0-baseline"
    )
    init.add_argument(
        "--database-url",
        default=None,
        help="覆盖 APP_DATABASE_URL，例如 sqlite:///./data/demo.db",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init-db":
        settings = Settings(database_url=args.database_url) if args.database_url else get_settings()
        summary = initialize_local_database(settings)
        print(
            "数据库初始化完成："
            f" model_types={summary['model_types']}"
            f" baselines={summary['baselines']}"
            f" checked={summary['created_or_existing_baselines']}"
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "initialize_local_database", "main"]
