import os
import inspect
import tempfile
import wave
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.core import task as pyannote_task
import whisper


def _extract_waveform_and_sr(audio: Any):
    if isinstance(audio, dict):
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        if waveform is None or sample_rate is None:
            raise ValueError("AUDIO input must contain 'waveform' and 'sample_rate'.")
        return waveform, int(sample_rate)

    if isinstance(audio, (list, tuple)) and len(audio) >= 2:
        return audio[0], int(audio[1])

    raise TypeError("Unsupported AUDIO format from ComfyUI.")


def _to_mono_float32_np(waveform: Any) -> np.ndarray:
    if isinstance(waveform, torch.Tensor):
        arr = waveform.detach().cpu().numpy()
    else:
        arr = np.asarray(waveform)

    arr = np.squeeze(arr)

    if arr.ndim == 1:
        mono = arr
    elif arr.ndim == 2:
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            mono = arr.mean(axis=0)
        else:
            mono = arr.mean(axis=1)
    else:
        raise ValueError(f"Unsupported waveform shape: {arr.shape}")

    mono = np.asarray(mono, dtype=np.float32)
    if mono.size == 0:
        raise ValueError("Waveform is empty.")

    max_abs = np.max(np.abs(mono))
    if max_abs > 1.0:
        mono = mono / 32768.0

    mono = np.clip(mono, -1.0, 1.0)
    return mono


def _write_wav(path: str, mono_audio: np.ndarray, sample_rate: int):
    pcm16 = (mono_audio * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def _is_effectively_silent(audio: np.ndarray, rms_threshold: float = 1e-4) -> bool:
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    return rms < rms_threshold


def _configure_torch_safe_globals():
    # PyTorch 2.6+ defaults torch.load(weights_only=True). Some pyannote
    # checkpoints include TorchVersion in metadata, so allowlist it.
    serialization = getattr(torch, "serialization", None)
    torch_version_mod = getattr(torch, "torch_version", None)
    torch_version_cls = getattr(torch_version_mod, "TorchVersion", None)
    add_safe_globals = getattr(serialization, "add_safe_globals", None)

    if add_safe_globals is None:
        return

    safe = []
    if torch_version_cls is not None:
        safe.append(torch_version_cls)

    # Allowlist only classes from pyannote.audio.core.task.
    # This avoids brittle one-by-one additions (Problem, Resolution, etc.)
    # while staying much tighter than broad package-level allowlisting.
    for name in dir(pyannote_task):
        value = getattr(pyannote_task, name, None)
        if isinstance(value, type) and getattr(value, "__module__", "") == pyannote_task.__name__:
            safe.append(value)

    if safe:
        add_safe_globals(safe)


def _load_pipeline_with_inspect_guard(token: str):
    # Work around recursion triggered by pytorch-lightning -> inspect.stack()
    # interacting with speechbrain lazy imports in some embedded envs.
    original_stack = inspect.stack

    def _safe_stack(context=0):
        return []

    inspect.stack = _safe_stack
    try:
        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
    finally:
        inspect.stack = original_stack


def _format_mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    mins = total // 60
    secs = total % 60
    return f"{mins}:{secs:02d}"


def _pick_speaker_for_segment(
    seg_start: float,
    seg_end: float,
    diar_turns: List[Tuple[float, float, str]],
):
    best_speaker = None
    best_overlap = 0.0

    for start, end, speaker in diar_turns:
        overlap = max(0.0, min(seg_end, end) - max(seg_start, start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker

    return best_speaker, best_overlap


class PyannoteDiarizationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "hf_token": ("STRING", {"multiline": False, "default": ""}),
                "whisper_model": (
                    [
                        "tiny",
                        "base",
                        "small",
                        "medium",
                        "large",
                        "turbo",
                    ],
                    {"default": "small"},
                ),
                "merge_consecutive_speaker": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("diarization_text",)
    FUNCTION = "run_diarization"
    CATEGORY = "audio"

    def run_diarization(self, audio, hf_token, whisper_model, merge_consecutive_speaker):
        token = (hf_token or "").strip()
        if not token:
            raise ValueError("HF token is required.")

        waveform, sample_rate = _extract_waveform_and_sr(audio)
        mono_audio = _to_mono_float32_np(waveform)
        if _is_effectively_silent(mono_audio):
            return ("",)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav_path = tmp.name

        try:
            _write_wav(tmp_wav_path, mono_audio, sample_rate)
            _configure_torch_safe_globals()

            pipeline = _load_pipeline_with_inspect_guard(token)
            diarization = pipeline(tmp_wav_path)

            diar_turns: List[Tuple[float, float, str]] = []
            speaker_order = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                diar_turns.append((float(turn.start), float(turn.end), str(speaker)))
                if speaker not in speaker_order:
                    speaker_order.append(speaker)

            if not diar_turns:
                return ("No speaker segments detected.",)

            total_speech = sum(max(0.0, end - start) for start, end, _ in diar_turns)
            if total_speech < 0.5:
                return ("",)

            speaker_to_letter = {
                speaker: chr(ord("A") + idx) for idx, speaker in enumerate(speaker_order)
            }

            device = "cuda" if torch.cuda.is_available() else "cpu"
            asr_model = whisper.load_model(whisper_model, device=device)
            asr = asr_model.transcribe(
                tmp_wav_path,
                verbose=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.7,
                logprob_threshold=-1.0,
            )
            asr_segments = asr.get("segments", []) or []
            if not asr_segments:
                return ("",)

            entries = []
            for segment in asr_segments:
                seg_start = float(segment.get("start", 0.0))
                seg_end = float(segment.get("end", seg_start))
                seg_duration = max(0.0, seg_end - seg_start)
                text = (segment.get("text") or "").strip()
                if not text:
                    continue
                if float(segment.get("no_speech_prob", 0.0)) >= 0.6:
                    continue
                if float(segment.get("avg_logprob", 0.0)) <= -1.2:
                    continue
                if seg_duration < 0.35:
                    continue

                speaker, overlap = _pick_speaker_for_segment(seg_start, seg_end, diar_turns)
                if speaker is None or overlap < 0.2:
                    speaker_label = "UNKNOWN"
                else:
                    speaker_label = f"SPEAKER {speaker_to_letter.get(speaker, '?')}"
                if speaker_label == "UNKNOWN":
                    continue

                entries.append((seg_start, speaker_label, text))

            if merge_consecutive_speaker and entries:
                merged = []
                cur_start, cur_speaker, cur_text = entries[0]
                for seg_start, speaker_label, text in entries[1:]:
                    if speaker_label == cur_speaker:
                        cur_text = f"{cur_text} {text}".strip()
                    else:
                        merged.append((cur_start, cur_speaker, cur_text))
                        cur_start, cur_speaker, cur_text = seg_start, speaker_label, text
                merged.append((cur_start, cur_speaker, cur_text))
                entries = merged

            lines = [f"{_format_mmss(seg_start)} {speaker_label}: {text}" for seg_start, speaker_label, text in entries]

            if not lines:
                result = ""
            else:
                result = "\n".join(lines)
            return (result,)
        finally:
            try:
                os.remove(tmp_wav_path)
            except OSError:
                pass


NODE_CLASS_MAPPINGS = {
    "PyannoteDiarizationNode": PyannoteDiarizationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyannoteDiarizationNode": "Speaker Diarization (pyannote)",
}
