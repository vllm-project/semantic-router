import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SKILL_CATALOG_PATH = (
    PROJECT_ROOT / "dashboard" / "backend" / "config" / "openclaw-skills.json"
)
SKILL_ROOT = PROJECT_ROOT / "skills"


def test_openclaw_vsr_bridge_skill_is_cataloged_and_packaged() -> None:
    catalog = json.loads(SKILL_CATALOG_PATH.read_text(encoding="utf-8"))

    bridge_entry = next(
        (item for item in catalog if item.get("id") == "openclaw-vsr-bridge"),
        None,
    )

    assert bridge_entry is not None
    assert bridge_entry["builtin"] is True
    assert bridge_entry["category"] == "development"
    assert "curl" in bridge_entry.get("requires", [])
    assert "bash" in bridge_entry.get("requires", [])

    skill_doc = SKILL_ROOT / "openclaw-vsr-bridge" / "SKILL.md"
    assert skill_doc.exists()

    skill_text = skill_doc.read_text(encoding="utf-8")
    assert "vllm-sr config import --from openclaw" in skill_text
    assert "--mode cli --runtime skip --no-launch" in skill_text
