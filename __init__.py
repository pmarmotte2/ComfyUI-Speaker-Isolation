from .speaker_isolation_nodes import SpeakerDiarizerNode

NODE_CLASS_MAPPINGS = {
    "SpeakerDiarizer": SpeakerDiarizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeakerDiarizer": "Speaker Diarizer (Isolation)",
}

WEB_DIRECTORY = "./js" # Optional: if you have custom JS for the UI

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("ComfyUI-Speaker-Isolation: Loaded SpeakerDiarizerNode")
