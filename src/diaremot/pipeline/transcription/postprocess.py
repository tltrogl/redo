from __future__ import annotations

from typing import Any, Iterable, List

from .models import TranscriptionSegment

__all__ = ["distribute_batch_results", "distribute_text_proportionally"]


def distribute_batch_results(
    batch_result: TranscriptionSegment,
    boundaries: Iterable[dict[str, Any]],
) -> list[TranscriptionSegment]:
    boundaries = list(boundaries)
    if not batch_result.words:
        return distribute_text_proportionally(batch_result, boundaries)

    results: List[TranscriptionSegment] = []
    for boundary in boundaries:
        seg = boundary["original"]
        concat_start = boundary["concat_start"]
        concat_end = boundary["concat_end"]

        segment_words: list[dict[str, Any]] = []
        segment_text_parts: list[str] = []

        for word in batch_result.words:
            word_start = word.get("start", 0.0)
            word_end = word.get("end", word_start)
            if word_start < concat_end and word_end > concat_start:
                adjusted = word.copy()
                time_offset = seg["start_time"] - concat_start
                adjusted["start"] = word_start + time_offset
                adjusted["end"] = word_end + time_offset
                segment_words.append(adjusted)

                word_text = word.get("word", "").strip()
                if word_text:
                    segment_text_parts.append(word_text)

        segment_text = " ".join(segment_text_parts).strip()
        if not segment_text:
            segment_text = "[silence]"

        results.append(
            TranscriptionSegment(
                start_time=seg["start_time"],
                end_time=seg["end_time"],
                text=segment_text,
                confidence=batch_result.confidence,
                speaker_id=seg.get("speaker_id"),
                speaker_name=seg.get("speaker_name"),
                words=segment_words if segment_words else None,
                language=batch_result.language,
                language_probability=batch_result.language_probability,
                model_used=f"{batch_result.model_used}-batched",
                asr_logprob_avg=batch_result.asr_logprob_avg,
            )
        )

    return results


def distribute_text_proportionally(
    batch_result: TranscriptionSegment,
    boundaries: Iterable[dict[str, Any]],
) -> list[TranscriptionSegment]:
    boundaries = list(boundaries)
    if not batch_result.text.strip():
        return []

    total_duration = sum(b["concat_end"] - b["concat_start"] for b in boundaries)
    words = batch_result.text.split()
    results: List[TranscriptionSegment] = []
    word_index = 0

    for boundary in boundaries:
        seg = boundary["original"]
        duration = boundary["concat_end"] - boundary["concat_start"]
        if total_duration <= 0:
            word_count = len(words) // len(boundaries) if boundaries else len(words)
        else:
            word_count = max(1, int(len(words) * (duration / total_duration)))
        segment_words = words[word_index : word_index + word_count]
        word_index += word_count

        segment_text = " ".join(segment_words)
        if not segment_text:
            segment_text = "[silence]"

        results.append(
            TranscriptionSegment(
                start_time=seg["start_time"],
                end_time=seg["end_time"],
                text=segment_text,
                confidence=batch_result.confidence,
                speaker_id=seg.get("speaker_id"),
                speaker_name=seg.get("speaker_name"),
                words=None,
                language=batch_result.language,
                language_probability=batch_result.language_probability,
                model_used=f"{batch_result.model_used}-distributed",
                asr_logprob_avg=batch_result.asr_logprob_avg,
            )
        )

    return results
