import sys
import torch
import torchaudio
import numpy as np
from comfy import model_management as mm

class SpeakerDiarizerChronoNode:
    """
    Speaker diarization for ComfyUI with guaranteed chronological ordering:
    speaker_1_audio = first person to speak, speaker_2_audio = second, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "hf_token": ("STRING", {"default": "", "multiline": False, "tooltip": "Hugging Face token for pyannote.audio"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto", "tooltip": "Compute device"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("speaker_1_audio", "speaker_2_audio", "speaker_3_audio", "speaker_4_audio", "summary")
    FUNCTION = "diarize_audio"
    CATEGORY = "Audio/Isolation"

    def _silent_outputs(self, audio, msg):
        sr = audio["sample_rate"]
        wf = audio["waveform"]
        samples = wf.shape[-1]
        silent = {"waveform": torch.zeros((1, 1, samples)), "sample_rate": sr}
        return silent, silent, silent, silent, msg

    def diarize_audio(self, audio, hf_token, device):
        # Make logs easy to identify this new node
        print("[SpeakerDiarizerChronoNode] Starting…")
        try:
            sys.setrecursionlimit(3000)
            print("[SpeakerDiarizerChronoNode] Recursion limit set to", sys.getrecursionlimit())
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception as e:
            print(f"[SpeakerDiarizerChronoNode] Warning: cannot restrict threads: {e}")

        # Device selection
        if device == "auto":
            processing_device = mm.get_torch_device()
        elif device == "cuda":
            processing_device = torch.device("cuda")
        else:
            processing_device = torch.device("cpu")

        # Prepare waveform
        sr = audio["sample_rate"]
        wf = audio["waveform"]
        print(f"[SpeakerDiarizerChronoNode] Original waveform {wf.shape} @ {sr}Hz")

        if wf.ndim == 3:      # (B, C, S)
            mono = wf[0].mean(dim=0)
        elif wf.ndim == 2:    # (C, S)
            mono = wf[0]
        else:                 # (S,)
            mono = wf
        mono = mono.cpu()

        target_sr = 16000
        if sr != target_sr:
            print(f"[SpeakerDiarizerChronoNode] Resampling {sr} → {target_sr}")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            mono = resampler(mono)

        audio_for_diar = {"waveform": mono.unsqueeze(0), "sample_rate": target_sr}
        print(f"[SpeakerDiarizerChronoNode] Waveform for diarization: {audio_for_diar['waveform'].shape}")

        # Diarization
        try:
            from pyannote.audio import Pipeline
            print(f"[SpeakerDiarizerChronoNode] Loading pyannote pipeline on {processing_device}")
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
            pipeline.to(processing_device)

            diarization = pipeline(audio_for_diar)
            print("[SpeakerDiarizerChronoNode] Diarization result:\n", diarization)
        except Exception as e:
            import traceback
            err = f"Error during diarization: {str(e)}\n{traceback.format_exc()}"
            print("[SpeakerDiarizerChronoNode]", err)
            return self._silent_outputs(audio, err)

        # Post-processing with strict chronological order
        try:
            # Collect segments per label (DO NOT use alphabetical label order anywhere)
            speaker_segments = {}
            for turn, _, label in diarization.itertracks(yield_label=True):
                speaker_segments.setdefault(label, []).append((float(turn.start), float(turn.end)))

            # Compute first start per speaker and sort
            speaker_first_start = {lab: min(s[0] for s in segs) for lab, segs in speaker_segments.items()}
            speakers_ordered = sorted(speaker_first_start.keys(), key=lambda k: speaker_first_start[k])

            print("[SpeakerDiarizerChronoNode] Chronological mapping (this defines output order):")
            for i, lab in enumerate(speakers_ordered, 1):
                print(f"  Output {i} -> {lab} @ {speaker_first_start[lab]:.2f}s")

            # Prepare original sample rate waveform for output building
            wf = audio["waveform"]
            if wf.ndim == 3:
                src = wf[0].mean(dim=0)
            elif wf.ndim == 2:
                src = wf[0]
            else:
                src = wf
            samples = src.shape[0]
            sr = audio["sample_rate"]

            outputs = []
            # Build tracks strictly following speakers_ordered
            for i, lab in enumerate(speakers_ordered[:4]):
                spk_wf = torch.zeros_like(src)
                for start_t, end_t in speaker_segments[lab]:
                    start = int(start_t * sr)
                    end = int(end_t * sr)
                    start = max(0, min(start, samples))
                    end = max(0, min(end, samples))
                    if end > start:
                        spk_wf[start:end] = src[start:end]
                outputs.append({"waveform": spk_wf.unsqueeze(0).unsqueeze(0), "sample_rate": sr})
                print(f"[SpeakerDiarizerChronoNode] Built output {i+1} for {lab}")

            # Pad remaining outputs with silence
            while len(outputs) < 4:
                outputs.append({"waveform": torch.zeros_like(src).unsqueeze(0).unsqueeze(0), "sample_rate": sr})

            summary_lines = [f"Output {i+1} -> {lab} @ {speaker_first_start[lab]:.2f}s"
                             for i, lab in enumerate(speakers_ordered)]
            summary = "Speakers ordered by first appearance:\n" + "\n".join(summary_lines)

            return tuple(outputs) + (summary,)

        except Exception as e:
            import traceback
            err = f"Error in postprocessing: {str(e)}\n{traceback.format_exc()}"
            print("[SpeakerDiarizerChronoNode]", err)
            return self._silent_outputs(audio, err)

# Register as a NEW node (new class key + new display name to avoid cache/old class issues)
NODE_CLASS_MAPPINGS = {
    "SpeakerDiarizerChronoNode": SpeakerDiarizerChronoNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeakerDiarizerChronoNode": "Speaker Diarizer (First Speaker = Output 1)"
}
