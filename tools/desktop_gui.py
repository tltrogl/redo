"""Simple desktop GUI for DiaRemot using PySimpleGUI.

This tool provides a minimal, no-server GUI to run the pipeline locally.
It prefers to import the pipeline run function (`run_pipeline`) and config
builder (`build_pipeline_config`) to run in-process, but will fall back to
launching `python -m diaremot.cli` as a subprocess if imports fail.

Usage:
    # (ensure venv activated and deps installed)
    pip install PySimpleGUI
    python tools/desktop_gui.py

Notes:
- The GUI is intentionally minimal: select input, output, choose `core` or `run`,
  optional flags, and Start. Progress and log output appear in the window.
- Long-running work runs in a background thread to keep the UI responsive.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import traceback
from pathlib import Path
from typing import Any

import PySimpleGUI as sg

logger = logging.getLogger("diaremot.gui")
logging.basicConfig(level=logging.INFO)

# Try to import runner functions for in-process execution
try:
    from diaremot.pipeline.run_pipeline import build_pipeline_config, run_pipeline  # type: ignore

    _IN_PROCESS = True
except Exception:
    build_pipeline_config = None  # type: ignore
    run_pipeline = None  # type: ignore
    _IN_PROCESS = False


def _run_in_process(
    input_path: str, output_dir: str, overrides: dict[str, Any], q: queue.Queue[str]
) -> None:
    try:
        q.put("Building config...")
        config = build_pipeline_config(overrides)  # type: ignore[arg-type]
        q.put("Starting pipeline (in-process)...")

        # Simple progress simulation since we can't easily hook into the pipeline's internal progress yet
        # In a real implementation, we'd pass a callback to run_pipeline if supported
        q.put("PROGRESS:10")

        result = run_pipeline(str(input_path), str(output_dir), config=config, clear_cache=False)  # type: ignore[call-arg]

        q.put("PROGRESS:100")
        q.put("Pipeline finished successfully.")
        q.put(repr(result))
    except Exception as exc:
        q.put(f"Pipeline failed: {exc}")
        tb = traceback.format_exc()
        q.put(tb)


def _run_subprocess(
    input_path: str, output_dir: str, command: str, extra_args: list[str], q: queue.Queue[str]
) -> None:
    # Build CLI args
    q.put("Running pipeline via subprocess...")
    py = os.environ.get("PYTHON", "python")
    args = [
        py,
        "-m",
        "diaremot.cli",
        command,
        "--input",
        str(input_path),
        "--outdir",
        str(output_dir),
    ] + extra_args
    q.put("Command: " + " ".join(args))
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout
        for line in proc.stdout:
            q.put(line.rstrip())
        rc = proc.wait()
        q.put(f"Process exited with code {rc}")
    except Exception as exc:
        q.put(f"Subprocess failed: {exc}")
        q.put(traceback.format_exc())


def worker_thread(
    input_path: str, output_dir: str, mode: str, overrides: dict[str, Any], q: queue.Queue[str]
):
    try:
        if _IN_PROCESS and build_pipeline_config is not None and run_pipeline is not None:
            _run_in_process(input_path, output_dir, overrides, q)
        else:
            # Convert overrides into CLI flags
            extra_args: list[str] = []
            for k, v in overrides.items():
                if isinstance(v, bool):
                    if v:
                        extra_args.append(f"--{k}")
                else:
                    extra_args.append(f"--{k}")
                    extra_args.append(str(v))
            _run_subprocess(input_path, output_dir, mode, extra_args, q)
    except Exception as exc:  # pragma: no cover - defensive
        q.put(f"Worker exception: {exc}")
        q.put(traceback.format_exc())


def make_layout() -> list[list[Any]]:
    layout = [
        [sg.Text("DiaRemot Desktop GUI", font=(None, 14))],
        [
            sg.Text("Input file", size=(10, 1)),
            sg.Input(key="-IN-", enable_events=True),
            sg.FileBrowse(file_types=(("Audio/Video", "*.wav;*.mp3;*.mp4;*.m4a;*.flac"),)),
        ],
        [sg.Text("Output dir", size=(10, 1)), sg.Input(key="-OUT-"), sg.FolderBrowse()],
        [
            sg.Text("Mode", size=(10, 1)),
            sg.Combo(["run", "core"], default_value="run", key="-MODE-"),
        ],
        [
            sg.Text("Speaker limit (opt)", size=(15, 1)),
            sg.Input(key="-SPEAKERS-", size=(6, 1)),
            sg.Checkbox("Disable affect", key="-NOAFFECT-"),
            sg.Checkbox("Disable SED", key="-NOSED-"),
            sg.Checkbox("Force Detailed SED", key="-FORCESED-", default=True),
        ],
        [sg.ProgressBar(100, orientation="h", size=(50, 20), key="-PROG-")],
        [
            sg.Button("Start", key="-START-"),
            sg.Button("Stop", key="-STOP-", disabled=True),
            sg.Button("Open Output", key="-OPEN-OUT-"),
        ],
        [sg.Text("Status log")],
        [sg.Multiline(key="-LOG-", size=(80, 20), autoscroll=True, disabled=True)],
    ]
    return layout


def run_gui():
    sg.theme("SystemDefault")
    layout = make_layout()
    window = sg.Window("DiaRemot Desktop", layout, finalize=True)

    q: queue.Queue[str] = queue.Queue()
    worker: threading.Thread | None = None

    try:
        while True:
            event, values = window.read(timeout=200)
            if event == sg.WIN_CLOSED:
                break

            # Poll queue for messages
            try:
                while True:
                    msg = q.get_nowait()
                    if msg.startswith("PROGRESS:"):
                        try:
                            val = int(msg.split(":")[1])
                            window["-PROG-"].update(val)
                        except ValueError:
                            pass
                        continue

                    current = window["-LOG-"].get()
                    window["-LOG-"].update(current + msg + "\n")
            except queue.Empty:
                pass

            if event == "-START-":
                in_file = values["-IN-"]
                out_dir = values["-OUT-"]
                if not in_file:
                    sg.popup_error("Please select an input file")
                    continue
                if not out_dir:
                    sg.popup_error("Please select an output directory")
                    continue
                # Prepare output dir
                out_path = Path(out_dir)
                out_path.mkdir(parents=True, exist_ok=True)

                mode = values["-MODE-"]
                overrides: dict[str, Any] = {}
                if values.get("-NOAFFECT-"):
                    overrides["disable_affect"] = True
                if values.get("-NOSED-"):
                    overrides["enable_sed"] = False
                if values.get("-FORCESED-"):
                    overrides["sed_mode"] = "timeline"
                sp_limit = values.get("-SPEAKERS-")
                if sp_limit:
                    try:
                        overrides["speaker_limit"] = int(sp_limit)
                    except ValueError:
                        sg.popup_error("speaker limit must be an integer")
                        continue

                # Disable Start button, enable Stop
                window["-START-"].update(disabled=True)
                window["-STOP-"].update(disabled=False)

                # Launch worker
                worker = threading.Thread(
                    target=worker_thread,
                    args=(in_file, str(out_path), mode, overrides, q),
                    daemon=True,
                )
                worker.start()
                q.put("Job started...")

            if event == "-STOP-":
                # Not trying to forcibly kill pipeline; just notify user
                q.put("Stop requested. If the pipeline supports cancellation, it will be honored.")
                window["-START-"].update(disabled=False)
                window["-STOP-"].update(disabled=True)

            if event == "-OPEN-OUT-":
                out_dir = values["-OUT-"]
                if out_dir and Path(out_dir).exists():
                    try:
                        if os.name == "nt":
                            os.startfile(out_dir)  # type: ignore
                        else:
                            subprocess.Popen(["xdg-open", out_dir])
                    except Exception as exc:
                        q.put(f"Failed to open output dir: {exc}")
                else:
                    sg.popup_error("Output directory does not exist")

            # If worker finished, re-enable Start
            if worker is not None and not worker.is_alive():
                window["-START-"].update(disabled=False)
                window["-STOP-"].update(disabled=True)
                worker = None

    finally:
        window.close()


if __name__ == "__main__":
    run_gui()
