# ComfyUI-Speaker-Isolation

A custom node for ComfyUI that performs speaker diarization to isolate individual speaker audio tracks from a single audio source.

## Features

-   Takes a single audio input.
-   Uses `pyannote.audio` for speaker diarization.
-   Outputs up to four separate audio tracks, one for each identified speaker.
-   Each output track maintains the original audio's full duration, with silence inserted where a specific speaker is not active.
-   Provides a summary of the diarization results (number of speakers, time per speaker).

## Node: Speaker Diarizer (Isolation)

-   **Category:** `Audio/Isolation`
-   **Inputs:**
    -   `audio` (AUDIO): The input audio file/data.
    -   `hf_token` (STRING): Your Hugging Face access token. This is **required** to download and use `pyannote.audio` pretrained models. You can get a token from [hf.co/settings/tokens](https://hf.co/settings/tokens).
    -   `device` (COMBO): The device to run the diarization model on (`auto`, `cuda`, `cpu`).
-   **Outputs:**
    -   `speaker_1_audio` (AUDIO): Audio track for the first detected speaker.
    -   `speaker_2_audio` (AUDIO): Audio track for the second detected speaker.
    -   `speaker_3_audio` (AUDIO): Audio track for the third detected speaker.
    -   `speaker_4_audio` (AUDIO): Audio track for the fourth detected speaker.
    -   `diarization_summary` (STRING): A text summary of the diarization process (e.g., number of speakers found, duration per speaker).

If fewer than four speakers are detected, the remaining speaker audio outputs will contain only silence. If an error occurs (e.g., missing token, model download issue), all audio outputs will be silent, and the error will be reported in the `diarization_summary`.

## Installation

1.  **Clone this repository:**
    Navigate to your `ComfyUI/custom_nodes/` directory and run:
    ```bash
    git clone <repository_url_for_ComfyUI-Speaker-Isolation>
    ```
    (Replace `<repository_url_for_ComfyUI-Speaker-Isolation>` with the actual URL once it's hosted.)

2.  **Install Dependencies:**
    Navigate into the cloned directory `ComfyUI/custom_nodes/ComfyUI-Speaker-Isolation/` and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `pyannote.audio` and its dependencies.
    You also need `ffmpeg` installed on your system, as `pyannote.audio` (and `torchaudio`) may rely on it for loading various audio formats.

3.  **Hugging Face Token and Model Agreement:**
    -   You **must** have a Hugging Face account.
    -   You **must** accept the user conditions for the models used by `pyannote.audio`'s default diarization pipeline. As of writing, these are:
        -   `pyannote/segmentation-3.0` ([link](https://hf.co/pyannote/segmentation-3.0))
        -   `pyannote/speaker-diarization-3.1` ([link](https://hf.co/pyannote/speaker-diarization-3.1))
        -   The underlying speaker embedding model, often `speechbrain/speaker-recognition-ecapa-tdnn` ([link](https://hf.co/speechbrain/speaker-recognition-ecapa-tdnn)) or similar.
        Visit these Hugging Face model pages and accept their terms.
    -   Provide your Hugging Face access token (with read permissions) to the `hf_token` input of the node. This token is primarily used for the *initial download* of the models.
    -   **Offline Usage & Model Caching:** Once the necessary `pyannote.audio` models are downloaded for the first time, they are stored in your local Hugging Face cache (usually located at `~/.cache/huggingface/hub` or a path defined by the `HF_HOME` environment variable). Subsequent runs of this node will use these cached models, allowing for offline operation regarding model access. While the `hf_token` input remains, it may not be actively used for network requests if all models are cached.
    -   **Advanced Offline Setup:** For users who need to manage models in a specific local directory completely separate from the standard Hugging Face cache (e.g., for air-gapped environments after an initial setup elsewhere), `pyannote.audio` does support loading pipelines from local paths. This involves manually downloading all required model and configuration files and structuring them correctly. For more details on this advanced scenario, please refer to the [pyannote.audio FAQ](https://github.com/pyannote/pyannote-audio/blob/develop/FAQ.md#can-i-use-gated-models-and-pipelines-offline). The current version of this node uses the Hugging Face model identifier string, relying on the cache mechanism.

4.  **Restart ComfyUI.**

## Usage

1.  Add the "Speaker Diarizer (Isolation)" node from the "Audio/Isolation" category.
2.  Connect an audio source to the `audio` input.
3.  Enter your Hugging Face token into the `hf_token` field.
4.  Connect the desired `speaker_X_audio` outputs to other audio nodes (e.g., Save Audio, audio input for other processes).
5.  The `diarization_summary` can be connected to a text display node to see the results.

## Troubleshooting
-   **`RecursionError: maximum recursion depth exceeded`**: This can sometimes occur with `pyannote.audio` or its dependencies (like `speechbrain`). The node attempts to mitigate this by increasing Python's recursion limit. If it persists, it might indicate a deeper environment or version conflict.
-   **Errors related to model downloads or `ffmpeg`**: Ensure you have a working internet connection, a valid Hugging Face token, have accepted model user agreements on Hugging Face, and that `ffmpeg` is correctly installed and accessible in your system's PATH.
-   **No speakers detected / Incorrect diarization**: The quality of diarization depends on the audio quality and the `pyannote.audio` model's capabilities. Background noise, very short utterances, or heavy overlap can affect performance.
