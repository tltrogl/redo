#!/usr/bin/env python3
"""Standalone CLI for the DiaRemot preprocessing stack."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf

from diaremot.pipeline.audio_preprocessing import AudioPreprocessor, PreprocessConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audio preprocessing with auto-chunking")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument(
        "--denoise", choices=["none", "spectral_sub_soft"], default="spectral_sub_soft"
    )
    parser.add_argument("--loudness-mode", choices=["asr", "broadcast"], default="asr")
    parser.add_argument(
        "--chunk-threshold",
        type=float,
        default=30.0,
        help="Auto-chunk threshold (minutes)",
    )
    parser.add_argument("--chunk-size", type=float, default=20.0, help="Chunk size (minutes)")
    parser.add_argument("--no-chunking", action="store_true", help="Disable auto-chunking")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    media_cache_dir = Path.cwd() / ".cache" / "video_audio"

    config = PreprocessConfig(
        target_sr=args.target_sr,
        denoise=args.denoise,
        loudness_mode=args.loudness_mode,
        auto_chunk_enabled=not args.no_chunking,
        chunk_threshold_minutes=args.chunk_threshold,
        chunk_size_minutes=args.chunk_size,
        media_cache_dir=str(media_cache_dir),
    )

    preprocessor = AudioPreprocessor(config)

    print(f"Processing {args.input}...")
    start_time = time.time()

    try:
        result = preprocessor.process_file(args.input)
        sf.write(args.output, result.audio, result.sample_rate)

        elapsed = time.time() - start_time
        duration = result.duration_s

        print(f"✓ Processing complete in {elapsed:.1f}s")
        print(f"  Output: {args.output}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Sample rate: {result.sample_rate} Hz")

        if result.health:
            print("  Audio Health:")
            print(f"    SNR: {result.health.snr_db:.1f} dB")
            print(f"    RMS: {result.health.rms_db:.1f} dB")
            print(f"    Est. LUFS: {result.health.est_lufs:.1f}")
            print(f"    Dynamic range: {result.health.dynamic_range_db:.1f} dB")
            print(f"    Silence ratio: {result.health.silence_ratio:.1%}")
            print(f"    Clipping detected: {result.health.clipping_detected}")
            if result.health.is_chunked and result.health.chunk_info:
                print(f"    Processed in chunks: {result.health.chunk_info['num_chunks']}")
        return 0
    except Exception as exc:  # pragma: no cover - CLI reporting
        print(f"✗ Processing failed: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
