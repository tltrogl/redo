import asyncio
import json
import os
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from diaremot.cli import _assemble_config, _default_outdir_for_input, core_run_pipeline


class LogStream(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))

    def flush(self):
        pass


class PipelineWorker(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, input_path: Path, outdir: Path, config: dict, clear_cache: bool = False):
        super().__init__()
        self.input_path = input_path
        self.outdir = outdir
        self.config = config
        self.clear_cache = clear_cache

    def run(self):
        try:
            # Create a new event loop for this thread to support asyncio components
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            print(f"Starting pipeline for {self.input_path}...")
            print(f"Output directory: {self.outdir}")
            manifest = core_run_pipeline(
                str(self.input_path),
                str(self.outdir),
                config=self.config,
                clear_cache=self.clear_cache,
            )
            self.finished_signal.emit(manifest)

            loop.close()
        except Exception as e:
            self.error_signal.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DiaRemot GUI")
        self.resize(600, 450)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- File Selection Section ---
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()

        # Input File
        input_layout = QHBoxLayout()
        input_label = QLabel("Input Audio:")
        input_label.setFixedWidth(80)
        self.input_edit = QLineEdit()
        self.input_btn = QPushButton("Browse")
        self.input_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.input_btn)
        file_layout.addLayout(input_layout)

        # Output Directory
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Dir:")
        output_label.setFixedWidth(80)
        self.output_edit = QLineEdit()
        self.output_btn = QPushButton("Browse")
        self.output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(self.output_btn)
        file_layout.addLayout(output_layout)

        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # --- Configuration Tabs ---
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: General
        self.tab_general = QWidget()
        self.init_general_tab()
        self.tabs.addTab(self.tab_general, "General")

        # Tab 2: ASR Settings
        self.tab_asr = QWidget()
        self.init_asr_tab()
        self.tabs.addTab(self.tab_asr, "ASR")

        # Tab 3: Diarization
        self.tab_diar = QWidget()
        self.init_diar_tab()
        self.tabs.addTab(self.tab_diar, "Diarization")

        # Tab 4: Advanced
        self.tab_adv = QWidget()
        self.init_adv_tab()
        self.tabs.addTab(self.tab_adv, "Advanced")

        # --- Action Buttons ---
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Pipeline")
        self.run_btn.clicked.connect(self.run_pipeline)
        self.run_btn.setMinimumHeight(40)
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)
        self.exit_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.exit_btn)
        main_layout.addLayout(btn_layout)

        # --- Progress & Logs ---
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate
        self.progress.hide()
        main_layout.addWidget(self.progress)

        self.details_btn = QPushButton("Show Logs")
        self.details_btn.setCheckable(True)
        self.details_btn.clicked.connect(self.toggle_logs)
        main_layout.addWidget(self.details_btn)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.hide()
        main_layout.addWidget(self.log_output)

        # Redirect stdout/stderr
        self.log_stream = LogStream()
        self.log_stream.text_written.connect(self.append_log)
        sys.stdout = self.log_stream
        sys.stderr = self.log_stream

    def init_general_tab(self):
        layout = QVBoxLayout()

        self.chk_affect = QCheckBox("Enable Affect Analysis (Emotion, Intent)")
        self.chk_affect.setChecked(True)
        layout.addWidget(self.chk_affect)

        self.chk_sed = QCheckBox("Enable Background Sound Event Detection (SED)")
        self.chk_sed.setChecked(True)
        layout.addWidget(self.chk_sed)

        self.chk_sed_timeline = QCheckBox("Force Detailed SED Timeline")
        self.chk_sed_timeline.setToolTip(
            "Force generation of SED timeline regardless of noise score"
        )
        layout.addWidget(self.chk_sed_timeline)

        self.chk_nr = QCheckBox("Gentle Noise Reduction")
        layout.addWidget(self.chk_nr)

        self.chk_clear_cache = QCheckBox("Clear Cache Before Run")
        self.chk_clear_cache.setToolTip(
            "Clear cached diarization/transcription data before running"
        )
        layout.addWidget(self.chk_clear_cache)

        self.chk_ignore_cache = QCheckBox("Ignore Transcription Cache")
        self.chk_ignore_cache.setToolTip("Re-run transcription even if cached results exist")
        layout.addWidget(self.chk_ignore_cache)

        self.chk_remote = QCheckBox("Allow Remote Model Downloads")
        self.chk_remote.setToolTip("Skip local-first routing and allow remote downloads")
        layout.addWidget(self.chk_remote)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Profile Preset:"))
        self.combo_profile = QComboBox()
        self.combo_profile.addItems(["default", "fast", "accurate", "offline"])
        layout.addWidget(self.combo_profile)

        layout.addStretch()
        self.tab_general.setLayout(layout)

    def init_asr_tab(self):
        layout = QVBoxLayout()

        # Model
        layout.addWidget(QLabel("Whisper Model:"))
        self.combo_model = QComboBox()
        self.combo_model.setEditable(True)
        self.combo_model.addItems(
            [
                "tiny.en",
                "base.en",
                "small.en",
                "medium.en",
                "large-v2",
                "large-v3",
            ]
        )
        self.combo_model.setToolTip("Select a model or type a path/model name")
        layout.addWidget(self.combo_model)

        # Compute Type
        layout.addWidget(QLabel("Compute Type:"))
        self.combo_compute = QComboBox()
        self.combo_compute.addItems(["int8", "float32", "int8_float16"])
        layout.addWidget(self.combo_compute)

        # Beam Size
        layout.addWidget(QLabel("Beam Size:"))
        self.spin_beam = QSpinBox()
        self.spin_beam.setRange(1, 10)
        self.spin_beam.setValue(1)
        layout.addWidget(self.spin_beam)

        # Temperature
        layout.addWidget(QLabel("Temperature:"))
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(0.0, 1.0)
        self.spin_temp.setSingleStep(0.1)
        self.spin_temp.setValue(0.0)
        layout.addWidget(self.spin_temp)

        self.chk_async_asr = QCheckBox("Enable Async Transcription")
        layout.addWidget(self.chk_async_asr)

        layout.addStretch()
        self.tab_asr.setLayout(layout)

    def init_diar_tab(self):
        layout = QVBoxLayout()

        # VAD Threshold
        layout.addWidget(QLabel("VAD Threshold (0.0 - 1.0):"))
        self.spin_vad = QDoubleSpinBox()
        self.spin_vad.setRange(0.0, 1.0)
        self.spin_vad.setSingleStep(0.05)
        self.spin_vad.setValue(0.35)
        layout.addWidget(self.spin_vad)

        # Min Speech
        layout.addWidget(QLabel("Min Speech Duration (s):"))
        self.spin_min_speech = QDoubleSpinBox()
        self.spin_min_speech.setRange(0.1, 5.0)
        self.spin_min_speech.setValue(0.8)
        layout.addWidget(self.spin_min_speech)

        # Min Silence
        layout.addWidget(QLabel("Min Silence Duration (s):"))
        self.spin_min_silence = QDoubleSpinBox()
        self.spin_min_silence.setRange(0.1, 5.0)
        self.spin_min_silence.setValue(0.8)
        layout.addWidget(self.spin_min_silence)

        # Speaker Limit
        layout.addWidget(QLabel("Speaker Limit (0 for auto):"))
        self.spin_speakers = QSpinBox()
        self.spin_speakers.setRange(0, 20)
        self.spin_speakers.setValue(0)
        layout.addWidget(self.spin_speakers)

        self.chk_cpu_diar = QCheckBox("Use CPU-Optimized Diarizer")
        layout.addWidget(self.chk_cpu_diar)

        # Use SED Timeline for Diarization
        self.chk_use_sed_for_diar = QCheckBox("Use SED Timeline for Diarization Splitting")
        self.chk_use_sed_for_diar.setToolTip(
            "When enabled, use background SED timeline events as speech regions for diarization"
        )
        layout.addWidget(self.chk_use_sed_for_diar)

        layout.addStretch()
        self.tab_diar.setLayout(layout)

    def init_adv_tab(self):
        layout = QVBoxLayout()

        self.chk_chunk = QCheckBox("Enable Auto-Chunking")
        self.chk_chunk.setChecked(True)
        layout.addWidget(self.chk_chunk)

        # Chunk Threshold
        layout.addWidget(QLabel("Chunk Threshold (minutes):"))
        self.spin_chunk_thresh = QDoubleSpinBox()
        self.spin_chunk_thresh.setRange(1.0, 300.0)
        self.spin_chunk_thresh.setValue(30.0)
        layout.addWidget(self.spin_chunk_thresh)

        # Chunk Size
        layout.addWidget(QLabel("Chunk Size (minutes):"))
        self.spin_chunk_size = QDoubleSpinBox()
        self.spin_chunk_size.setRange(1.0, 300.0)
        self.spin_chunk_size.setValue(20.0)
        layout.addWidget(self.spin_chunk_size)

        # Chunk Overlap
        layout.addWidget(QLabel("Chunk Overlap (seconds):"))
        self.spin_chunk_overlap = QDoubleSpinBox()
        self.spin_chunk_overlap.setRange(0.0, 300.0)
        self.spin_chunk_overlap.setValue(30.0)
        layout.addWidget(self.spin_chunk_overlap)

        layout.addStretch()
        layout.addWidget(QLabel("SED Timeline Parameters"))

        # SED Enter / Exit thresholds
        layout.addWidget(QLabel("SED Enter Threshold (0.0 - 1.0):"))
        self.spin_sed_enter = QDoubleSpinBox()
        self.spin_sed_enter.setRange(0.0, 1.0)
        self.spin_sed_enter.setSingleStep(0.05)
        self.spin_sed_enter.setValue(0.50)
        layout.addWidget(self.spin_sed_enter)

        layout.addWidget(QLabel("SED Exit Threshold (0.0 - 1.0):"))
        self.spin_sed_exit = QDoubleSpinBox()
        self.spin_sed_exit.setRange(0.0, 1.0)
        self.spin_sed_exit.setSingleStep(0.05)
        self.spin_sed_exit.setValue(0.35)
        layout.addWidget(self.spin_sed_exit)

        layout.addWidget(QLabel("SED Median Kernel (odd integer >=1):"))
        self.spin_sed_median_k = QSpinBox()
        self.spin_sed_median_k.setRange(1, 51)
        self.spin_sed_median_k.setSingleStep(2)
        self.spin_sed_median_k.setValue(5)
        layout.addWidget(self.spin_sed_median_k)

        layout.addWidget(QLabel("SED Min Dur (default seconds or JSON label map):"))
        self.edit_sed_min_dur = QLineEdit()
        self.edit_sed_min_dur.setPlaceholderText("e.g. 0.3 or speech=0.5,music=1.0")
        layout.addWidget(self.edit_sed_min_dur)

        layout.addWidget(QLabel("SED Merge Gap (seconds):"))
        self.spin_sed_merge_gap = QDoubleSpinBox()
        self.spin_sed_merge_gap.setRange(0.0, 60.0)
        self.spin_sed_merge_gap.setValue(0.20)
        layout.addWidget(self.spin_sed_merge_gap)

        self.tab_adv.setLayout(layout)

    def toggle_logs(self):
        if self.details_btn.isChecked():
            self.log_output.show()
            self.details_btn.setText("Hide Logs")
        else:
            self.log_output.hide()
            self.details_btn.setText("Show Logs")

    def browse_input(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.m4a *.flac *.ogg)"
        )
        if fname:
            self.input_edit.setText(fname)

    def browse_output(self):
        dirname = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dirname:
            self.output_edit.setText(dirname)

    def append_log(self, text):
        self.log_output.moveCursor(self.log_output.textCursor().MoveOperation.End)
        self.log_output.insertPlainText(text)
        self.log_output.moveCursor(self.log_output.textCursor().MoveOperation.End)

    def run_pipeline(self):
        input_file = self.input_edit.text()
        if not input_file:
            QMessageBox.critical(self, "Error", "Please select an input file.")
            return

        input_path = Path(input_file)
        if not input_path.exists():
            QMessageBox.critical(self, "Error", "Input file does not exist.")
            return

        outdir_val = self.output_edit.text()
        if not outdir_val:
            outdir = _default_outdir_for_input(input_path)
        else:
            outdir = Path(outdir_val)

        # Build config overrides
        overrides = {
            # General
            "disable_affect": not self.chk_affect.isChecked(),
            "enable_sed": self.chk_sed.isChecked(),
            "noise_reduction": self.chk_nr.isChecked(),
            "sed_mode": "timeline" if self.chk_sed_timeline.isChecked() else "auto",
            "ignore_tx_cache": self.chk_ignore_cache.isChecked(),
            "local_first": not self.chk_remote.isChecked(),
            "cpu_diarizer": self.chk_cpu_diar.isChecked(),
            # ASR
            "whisper_model": self.combo_model.currentText(),
            "compute_type": self.combo_compute.currentText(),
            "beam_size": self.spin_beam.value(),
            "temperature": self.spin_temp.value(),
            # Diarization
            "vad_threshold": self.spin_vad.value(),
            "vad_min_speech_sec": self.spin_min_speech.value(),
            "vad_min_silence_sec": self.spin_min_silence.value(),
            "speaker_limit": self.spin_speakers.value() if self.spin_speakers.value() > 0 else None,
            "diar_use_sed_timeline": self.chk_use_sed_for_diar.isChecked(),
            # Advanced
            "chunk_enabled": self.chk_chunk.isChecked(),
            "chunk_threshold_minutes": self.spin_chunk_thresh.value(),
            "chunk_size_minutes": self.spin_chunk_size.value(),
            "chunk_overlap_seconds": self.spin_chunk_overlap.value(),
            # SED timeline parameters
            "sed_enter": float(self.spin_sed_enter.value()),
            "sed_exit": float(self.spin_sed_exit.value()),
            "sed_median_k": int(self.spin_sed_median_k.value()),
            "sed_merge_gap": float(self.spin_sed_merge_gap.value()),
            # Parse SED min duration mapping (label=seconds,comma separated) for GUI
            "sed_min_dur": (
                lambda t: None
                if not t
                else (
                    {
                        kv.split("=")[0].strip(): float(kv.split("=")[1])
                        for kv in t.split(",")
                        if "=" in kv
                    }
                    if "=" in t
                    else (
                        json.loads(t)
                        if (t.strip().startswith("{") and t.strip().endswith("}"))
                        else None
                    )
                )
            )(self.edit_sed_min_dur.text()),
        }

        if self.chk_sed_timeline.isChecked():
            overrides["enable_sed"] = True

        profile_name = self.combo_profile.currentText()
        config = _assemble_config(profile_name, overrides)

        # UI Updates
        self.run_btn.setEnabled(False)
        self.progress.show()
        self.log_output.clear()

        # Start Thread
        self.worker = PipelineWorker(
            input_path,
            outdir,
            config,
            clear_cache=self.chk_clear_cache.isChecked(),
        )
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_finished(self, manifest):
        self.run_btn.setEnabled(True)
        self.progress.hide()
        # Try to find the output directory in manifest with fallback keys
        output_dir = (
            manifest.get("output_dir")
            or manifest.get("outdir")
            or manifest.get("output")
            or self.output_edit.text()
        )
        if not output_dir:
            # Fallback to the default output dir for the input file
            input_file = self.input_edit.text()
            if input_file:
                output_dir = str(_default_outdir_for_input(Path(input_file)))
            else:
                output_dir = "unknown"
        QMessageBox.information(
            self,
            "Success",
            f"Pipeline Completed!\nOutput in: {output_dir}",
        )

    def on_error(self, error_msg):
        self.run_btn.setEnabled(True)
        self.progress.hide()
        QMessageBox.critical(self, "Error", f"Pipeline Failed:\n{error_msg}")

    def closeEvent(self, a0):
        # Restore stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if a0 is not None and hasattr(a0, "accept"):
            try:
                a0.accept()
            except Exception:
                # Defensive: ignore any accept errors
                pass


def main():
    # Force disable hf_transfer to prevent errors if the package is missing
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    # Diagnostics: print debug info to stdout so CLI logs capture startup activity
    print("DiaRemot GUI: starting QApplication")
    # Also write to a timestamped GUI startup log for situations where stdout may be hidden
    log_file = None
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "gui_startup.log"
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(f"DiaRemot GUI starting, pid={os.getpid()}\n")
    except Exception:
        pass
    try:
        app = QApplication(sys.argv)
        print("DiaRemot GUI: QApplication created")
        try:
            if log_file:
                with open(log_file, "a", encoding="utf-8") as fh:
                    fh.write("DiaRemot GUI: QApplication created\n")
        except Exception:
            pass
        print(f"DiaRemot GUI: pid={os.getpid()}, python={sys.executable}")
        # Attempt to collect GUI platform information
        try:
            from PyQt6.QtGui import QGuiApplication

            screens = QGuiApplication.screens()
            screen_count = len(screens)
            print(f"DiaRemot GUI: detected {screen_count} screen(s)")
            try:
                if log_file:
                    with open(log_file, "a", encoding="utf-8") as fh:
                        fh.write(f"DiaRemot GUI: detected {screen_count} screen(s)\n")
            except Exception:
                pass
            if screen_count == 0:
                print(
                    "DiaRemot GUI: No display screens detected. GUI may not be visible in a headless environment."
                )
        except Exception:
            pass
    except Exception as exc:
        print(f"DiaRemot GUI: failed to create QApplication: {exc}")
        raise

    try:
        window = MainWindow()
        print("DiaRemot GUI: MainWindow instantiated")
        try:
            if log_file:
                with open(log_file, "a", encoding="utf-8") as fh:
                    fh.write("DiaRemot GUI: MainWindow instantiated\n")
        except Exception:
            pass
        window.show()
        # Attempt to bring the window to the front on platforms where it may be hidden.
        try:
            window.raise_()
            window.activateWindow()
        except Exception:
            pass
        print("DiaRemot GUI: showing window")
        # Check visibility and write to log
        try:
            vis = window.isVisible()
            print(f"DiaRemot GUI: window visible? {vis}")
            if log_file and vis:
                with open(log_file, "a", encoding="utf-8") as fh:
                    fh.write(f"DiaRemot GUI: window visible? {vis}\n")
        except Exception:
            pass
        try:
            if log_file:
                with open(log_file, "a", encoding="utf-8") as fh:
                    fh.write("DiaRemot GUI: showing window\n")
        except Exception:
            pass
    except Exception as exc:
        print(f"DiaRemot GUI: failed to instantiate/show MainWindow: {exc}")
        raise

    try:
        sys.exit(app.exec())
    except Exception as exc:  # pragma: no cover - defensive GUI runtime
        print(f"DiaRemot GUI: QApplication execution failed: {exc}")
        try:
            if log_file:
                with open(log_file, "a", encoding="utf-8") as fh:
                    fh.write(f"DiaRemot GUI: QApplication execution failed: {exc}\n")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
