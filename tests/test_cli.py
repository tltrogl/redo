"""Regression tests for the DiaRemot Typer CLI."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from diaremot import cli


def test_enrich_command_returns_manifest(monkeypatch, tmp_path):
    """Ensure `diaremot enrich` surfaces the manifest without crashing."""

    audio_file = tmp_path / "input.wav"
    audio_file.write_bytes(b"fake")
    outdir = tmp_path / "outs"

    expected_manifest = {"run_id": "test", "out_dir": str(outdir)}
    captured_args: dict[str, object] = {}

    def _fake_run_pipeline(input_path: str, output_dir: str, *, config: dict[str, object]):
        captured_args["input_path"] = input_path
        captured_args["output_dir"] = output_dir
        captured_args["config"] = config
        return expected_manifest

    monkeypatch.setattr(cli, "core_run_pipeline", _fake_run_pipeline)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "enrich",
            str(audio_file),
            "--outdir",
            str(outdir),
        ],
    )

    assert result.exit_code == 0, result.stdout

    payload = json.loads(result.stdout.strip())
    assert payload == expected_manifest

    assert captured_args["input_path"] == str(audio_file)
    assert captured_args["output_dir"] == str(outdir)
    config = captured_args["config"]
    assert isinstance(config, dict)
    assert config["enable_sed"] is True
    assert config["disable_affect"] is False
    assert config["ignore_tx_cache"] is False
