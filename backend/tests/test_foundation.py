import tempfile
import unittest
from pathlib import Path

from sqlalchemy import text

from app.core.config import Settings
from app.db.database import create_engine_from_settings
from app.main import app


class FoundationTests(unittest.TestCase):
    def test_fastapi_entrypoint_is_importable(self) -> None:
        self.assertEqual(app.title, "模型训练可视化平台")
        self.assertTrue(any(route.path == "/api/health" for route in app.routes))

    def test_settings_provide_sqlite_and_storage_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            settings = Settings(
                database_url=f"sqlite:///{(tmp_path / 'platform.db').as_posix()}",
                data_dir=tmp_path / "data",
                model_dir=tmp_path / "models",
                script_dir=tmp_path / "scripts",
                upload_dir=tmp_path / "uploads",
            )

            self.assertTrue(settings.database_url.startswith("sqlite:///"))
            settings.ensure_directories()
            self.assertTrue(settings.data_dir.is_dir())
            self.assertTrue(settings.model_dir.is_dir())
            self.assertTrue(settings.script_dir.is_dir())
            self.assertTrue(settings.upload_dir.is_dir())

    def test_configured_sqlite_database_can_connect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_file = Path(directory) / "nested" / "platform.db"
            settings = Settings(database_url=f"sqlite:///{database_file.as_posix()}")
            engine = create_engine_from_settings(settings)

            with engine.connect() as connection:
                self.assertEqual(
                    connection.execute(text("SELECT 1")).scalar_one(),
                    1,
                )

            engine.dispose()

    def test_required_data_science_dependencies_are_importable(self) -> None:
        import joblib
        import numpy
        import pandas
        import sklearn

        self.assertTrue(joblib.__version__)
        self.assertTrue(numpy.__version__)
        self.assertTrue(pandas.__version__)
        self.assertTrue(sklearn.__version__)


if __name__ == "__main__":
    unittest.main()
