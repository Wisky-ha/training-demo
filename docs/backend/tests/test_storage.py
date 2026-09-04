from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.core.config import Settings
from backend.app.db.models import FileArtifactORM
from backend.app.db.session import create_session_factory, initialize_database
from backend.app.storage import ArtifactType, ArtifactNotFoundError, FileStorageService


@pytest.fixture
def storage_context(tmp_path):
    root = tmp_path / "isolated-storage"
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_database(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        yield FileStorageService(root, session=session), session, root
        session.rollback()
    engine.dispose()


def test_settings_can_select_an_isolated_artifact_root(tmp_path):
    root = tmp_path / "configured-root"
    settings = Settings(storage_root=root)
    storage = FileStorageService(settings)

    artifact = storage.save_script_source("configured-script", "pass")

    assert (root / artifact.relative_path).read_text(encoding="utf-8") == "pass"
    assert not (tmp_path / "data").exists()


def test_model_preprocessor_and_script_round_trip_with_metadata(storage_context):
    storage, session, root = storage_context
    model = storage.save_model("model-v1", b"model bytes")
    preprocessor = storage.save_preprocessor_state("model-v1", b"state bytes")
    script = storage.save_script_source("script-v1", "print('ok')\n")

    assert storage.read_model("model-v1") == b"model bytes"
    assert storage.read_preprocessor_state("model-v1") == b"state bytes"
    assert storage.read_script_source("script-v1") == "print('ok')\n"
    assert storage.model_exists("model-v1")
    assert storage.preprocessor_exists("model-v1")
    assert storage.script_exists("script-v1")

    for artifact in (model, preprocessor, script):
        path = Path(artifact.relative_path)
        assert not path.is_absolute()
        assert artifact.size_bytes == len((root / path).read_bytes())
        assert len(artifact.checksum_sha256) == 64
        assert artifact.checksum_sha256 == __import__("hashlib").sha256(
            (root / path).read_bytes()
        ).hexdigest()
        assert artifact.artifact_type in ArtifactType
        assert artifact.artifact_id

    records = session.scalars(select(FileArtifactORM)).all()
    assert len(records) == 3
    model_record = session.scalar(
        select(FileArtifactORM).where(
            FileArtifactORM.artifact_type == ArtifactType.MODEL.value,
            FileArtifactORM.artifact_id == "model-v1",
        )
    )
    assert model_record is not None
    assert model_record.relative_path == model.relative_path
    assert model_record.size_bytes == model.size_bytes
    assert model_record.checksum_sha256 == model.checksum_sha256


def test_repeated_save_replaces_content_and_metadata_without_duplicates(storage_context):
    storage, session, _ = storage_context
    first = storage.save_model("same-model", b"first")
    second = storage.save_model("same-model", b"second content")

    assert storage.read_model("same-model") == b"second content"
    assert second.relative_path == first.relative_path
    assert second.size_bytes == len(b"second content")
    assert second.checksum_sha256 != first.checksum_sha256
    assert len(session.scalars(select(FileArtifactORM)).all()) == 1


def test_path_traversal_identifiers_are_rejected_and_temp_root_is_isolated(storage_context, tmp_path):
    storage, _, root = storage_context
    for invalid in ("../outside", "nested/name", r"nested\\name", "/absolute", "", ".", ".."):
        with pytest.raises(ValueError):
            storage.save_model(invalid, b"blocked")

    outside = tmp_path / "outside.bin"
    assert not outside.exists()
    assert root.exists()
    assert not (tmp_path / "outside").exists()


def test_missing_files_are_reported_as_missing(storage_context):
    storage, _, root = storage_context
    artifact = storage.save_model("to-remove", b"payload")
    (root / artifact.relative_path).unlink()

    assert not storage.model_exists("to-remove")
    with pytest.raises(ArtifactNotFoundError):
        storage.read_model("to-remove")
