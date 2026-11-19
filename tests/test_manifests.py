"""Every run manifest records the av2net version that made it."""

from __future__ import annotations

import json
from pathlib import Path

import run_common
from semantic_pipeline import __version__


def test_a_manifest_is_headed_by_the_av2net_version(tmp_path):
    path = tmp_path / "run_manifest.example.json"
    run_common.write_manifest(path, {"layer": 2, "step": "example"})
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert list(manifest)[:3] == ["av2net_version", "layer", "step"]
    assert manifest["av2net_version"] == __version__


def test_every_run_script_writes_its_manifest_with_the_version():
    scripts = sorted(p for p in Path(run_common.__file__).parent.glob("run_*.py")
                     if p.name != "run_common.py")
    assert len(scripts) >= 9
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        assert "write_manifest(" in source, script.name
        writes = [line.strip() for line in source.splitlines() if "write_json(" in line]
        assert not [line for line in writes if "manifest" in line], script.name
