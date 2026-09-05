"""Upload the checked-in demo CSV and scripts to a running backend.

Requires the development dependencies (notably httpx). Existing script rows
with the same name/type/version are reused; each invocation uploads a fresh
CSV because the API intentionally treats uploads as immutable dataset records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


HERE = Path(__file__).resolve().parent
MODEL_TYPE = "electric_load"


def _script(
    client: httpx.Client, path: Path, name: str, script_type: str, version: str
) -> str:
    response = client.get("/api/scripts", params={"status": "ENABLED", "page_size": 100})
    response.raise_for_status()
    for item in response.json().get("items", []):
        if (
            item["name"] == name
            and item["script_type"] == script_type
            and item["version"] == version
        ):
            return item["id"]

    with path.open("rb") as source:
        response = client.post(
            "/api/scripts/upload",
            data={
                "name": name,
                "version": version,
                "script_type": script_type,
                "supported_model_types": json.dumps([MODEL_TYPE]),
            },
            files={"file": (path.name, source, "text/x-python")},
        )
    response.raise_for_status()
    return response.json()["id"]


def seed(base_url: str) -> dict[str, str]:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0) as client:
        preprocessor_id = _script(
            client,
            HERE / "scripts" / "demo_preprocessor.py",
            "demo-preprocessor",
            "preprocessor",
            "v1",
        )
        trainer_id = _script(
            client,
            HERE / "scripts" / "demo_trainer.py",
            "demo-trainer",
            "trainer",
            "v1",
        )
        csv_path = HERE / "energy_demo.csv"
        with csv_path.open("rb") as source:
            response = client.post(
                "/api/datasets/upload",
                files={"file": (csv_path.name, source, "text/csv")},
            )
        response.raise_for_status()
        dataset_id = response.json()["id"]
    return {
        "dataset_id": dataset_id,
        "preprocessor_id": preprocessor_id,
        "trainer_id": trainer_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="写入内部演示数据和脚本")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    print(json.dumps(seed(args.base_url), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
