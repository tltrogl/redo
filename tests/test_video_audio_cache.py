from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from diaremot.utils.video_audio_cache import ensure_cached_audio, is_probably_video


def test_is_probably_video_variants() -> None:
    assert is_probably_video("movie.mp4")
    assert is_probably_video("clip.MKV")
    assert not is_probably_video("speech.wav")
    assert not is_probably_video("audio.flac")


def test_ensure_cached_audio_crates_once(tmp_path: Path) -> None:
    video = tmp_path / "recording.MP4"
    video.write_text("dummy video content")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    def fake_run(cmd, check, stdout, stderr):
        tmp_wav_path = Path(cmd[-1])
        tmp_wav_path.write_bytes(b"RIFF\0\0\0")
        return subprocess.CompletedProcess(cmd, 0)

    with patch("diaremot.utils.video_audio_cache.subprocess.run", side_effect=fake_run):
        first = ensure_cached_audio(
            video,
            cache_dir=cache_dir,
            target_sr=16000,
            channels=1,
        )
        assert first.exists()
        second = ensure_cached_audio(
            video,
            cache_dir=cache_dir,
            target_sr=16000,
            channels=1,
        )
        assert second == first