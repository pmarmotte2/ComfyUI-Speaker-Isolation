# ComfyUI Speaker Diarization Node (pyannote)

Custom ComfyUI node using `pyannote-audio` speaker diarization.

## Inputs
- `audio` (`AUDIO`): ComfyUI audio input.
- `hf_token` (`STRING`): Hugging Face access token.
- `whisper_model`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large`, `turbo`).
- `merge_consecutive_speaker` (`BOOLEAN`): merge consecutive lines when the same speaker keeps talking.

## Output
- `diarization_text` (`STRING`) in format:
  - `0:10 SPEAKER A: ...`
  - `0:20 SPEAKER B: ...`

Each line is a Whisper transcription segment tagged with the speaker that overlaps most with the segment.

## Install
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Restart ComfyUI.

## Notes
- You must have access to `pyannote/speaker-diarization-3.1` on Hugging Face with your token.
- First run will download both pyannote and Whisper models.
