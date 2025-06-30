import sys # Added for recursion limit
import torch
import torchaudio
import numpy as np
import folder_paths
from comfy import model_management as mm

# Tentative import, will confirm during pyannote integration
# from pyannote.audio import Pipeline

class SpeakerDiarizerNode:
    """
    A ComfyUI node that performs speaker diarization on an audio input.
    It uses pyannote.audio to identify speaker segments and then outputs
    separate audio tracks for each identified speaker (up to 4), where each track
    maintains the original audio's duration by filling non-speaker parts with silence.
    """
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the SpeakerDiarizerNode.
        - audio: The input audio data.
        - hf_token: Hugging Face token for pyannote.audio models.
        - device: The device (auto, cuda, cpu) to run diarization on.
        """
        return {
            "required": {
                "audio": ("AUDIO",),
                "hf_token": ("STRING", {"default": "", "multiline": False, "tooltip": "Hugging Face access token for pyannote.audio models."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto", "tooltip": "Device to run the diarization model on."}),
            },
            "optional": {
                # Parameters for pyannote.audio.Pipeline if needed, e.g.
                # "min_speakers": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                # "max_speakers": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("speaker_1_audio", "speaker_2_audio", "speaker_3_audio", "speaker_4_audio", "diarization_summary")
    FUNCTION = "diarize_audio"
    CATEGORY = "Audio/Isolation" # Changed category

    def return_silent_outputs_with_message(self, original_audio_struct, message_string):
        """Helper to return 4 silent audio outputs and a message string in case of errors."""
        input_waveform = original_audio_struct["waveform"]
        input_sample_rate = original_audio_struct["sample_rate"]

        # Determine a representative mono waveform for duration
        if input_waveform.ndim == 3: # B, C, S
            mono_ref_waveform = input_waveform[0].mean(dim=0) if input_waveform.shape[1] > 1 else input_waveform[0,0]
        elif input_waveform.ndim == 2: # B, S
             mono_ref_waveform = input_waveform[0]
        else: # S
             mono_ref_waveform = input_waveform

        original_duration_samples = mono_ref_waveform.shape[0]

        silent_outputs = []
        for _ in range(4):
            silent_waveform_tensor = torch.zeros(original_duration_samples, dtype=mono_ref_waveform.dtype, device=mono_ref_waveform.device)
            silent_outputs.append({"waveform": silent_waveform_tensor.unsqueeze(0).unsqueeze(0), "sample_rate": input_sample_rate})
        return tuple(silent_outputs) + (message_string,)

    def diarize_audio(self, audio, hf_token, device): #, min_speakers=1, max_speakers=5):
        """
        Performs speaker diarization on the input audio.

        Args:
            audio (dict): The input audio in ComfyUI's AUDIO format
                          {"waveform": tensor, "sample_rate": int}.
            hf_token (str): Hugging Face access token for pyannote.audio.
            device (str): Target device for computation ("auto", "cuda", "cpu").
            # min_speakers (int, optional): Minimum number of speakers.
            # max_speakers (int, optional): Maximum number of speakers.

        Returns:
            tuple: A tuple containing:
                - speaker_1_audio (AUDIO): Audio for speaker 1, or None/silence.
                - speaker_2_audio (AUDIO): Audio for speaker 2, or None/silence.
                - speaker_3_audio (AUDIO): Audio for speaker 3, or None/silence.
                - speaker_4_audio (AUDIO): Audio for speaker 4, or None/silence.
                - diarization_summary (STRING): A summary of the diarization results.
        """
        # Attempt to increase recursion limit as a potential workaround for deep library calls
        try:
            current_recursion_limit = sys.getrecursionlimit()
            # Setting it to a common higher value. Might need adjustment.
            # Only set if it's lower than a reasonable threshold to avoid repeatedly setting it unnecessarily
            # or if we want to enforce a specific higher limit.
            # For now, let's set it if it's below, e.g., 3000.
            if current_recursion_limit < 3000:
                 sys.setrecursionlimit(3000)
            print(f"SpeakerDiarizerNode: Recursion limit set to {sys.getrecursionlimit()} (was {current_recursion_limit})")
        except Exception as e_recursion:
            print(f"SpeakerDiarizerNode: Warning - Could not set recursion limit: {e_recursion}")

        # --- Device Selection ---
        if device == "auto":
            processing_device = mm.get_torch_device()
        elif device == "cuda":
            processing_device = torch.device("cuda")
        else:
            processing_device = torch.device("cpu")

        # --- Input Audio Preparation ---
        input_sample_rate = audio["sample_rate"]
        raw_waveform_tensor = audio["waveform"] # Expected (B, C, S) by ComfyUI standard

        print(f"SpeakerDiarizerNode: Original input waveform shape: {raw_waveform_tensor.shape}, sample rate: {input_sample_rate}")

        # Take the first item in the batch
        if raw_waveform_tensor.ndim < 2: # Should not happen with ComfyUI AUDIO spec
            print(f"SpeakerDiarizerNode: Warning - input waveform has unexpected ndim < 2: {raw_waveform_tensor.shape}. Attempting to process.")
            # If it's (S,), ensure it's on CPU for resampler
            mono_waveform_for_prep = raw_waveform_tensor.cpu()
        elif raw_waveform_tensor.ndim == 2: # Potentially (B,S) or (C,S) if B=1 was squeezed by user
             # If first dim is small (likely channel or batch=1) and second is large (samples)
            if raw_waveform_tensor.shape[0] == 1: # (1,S) - mono already, just remove batch
                mono_waveform_for_prep = raw_waveform_tensor.squeeze(0).cpu()
            elif raw_waveform_tensor.shape[0] > 1 and raw_waveform_tensor.shape[0] < raw_waveform_tensor.shape[1]: # (C,S) multi-channel, no batch
                mono_waveform_for_prep = torch.mean(raw_waveform_tensor, dim=0).cpu() # Average channels
            else: # Assume (B,S) where B > 1, take first batch
                mono_waveform_for_prep = raw_waveform_tensor[0].cpu()
        else: # Expected (B, C, S)
            current_waveform_b0 = raw_waveform_tensor[0] # Shape (C, S) from first batch item
            # Convert to mono by averaging channels if multi-channel
            if current_waveform_b0.shape[0] > 1: # If C > 1
                mono_waveform_for_prep = torch.mean(current_waveform_b0, dim=0).cpu() # Shape (S,)
            else: # C == 1
                mono_waveform_for_prep = current_waveform_b0.squeeze(0).cpu() # Shape (S,)

        print(f"SpeakerDiarizerNode: Prepared mono waveform for resampling, shape: {mono_waveform_for_prep.shape}")

        # Resample if necessary
        target_sr = 16000
        if input_sample_rate != target_sr:
            print(f"SpeakerDiarizerNode: Resampling from {input_sample_rate}Hz to {target_sr}Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=input_sample_rate, new_freq=target_sr)
            diarization_input_tensor = resampler(mono_waveform_for_prep) # Shape (S,)
        else:
            diarization_input_tensor = mono_waveform_for_prep # Shape (S,)

        print(f"SpeakerDiarizerNode: Waveform for diarization (after potential resampling), shape: {diarization_input_tensor.shape}")

        # Pyannote expects (channel, time) for the waveform value in the dict.
        # So, unsqueeze to (1, S) for mono.
        diarization_input_tensor_for_pipeline = diarization_input_tensor.unsqueeze(0) # Shape (1,S)

        print(f"SpeakerDiarizerNode: Final waveform shape for pyannote pipeline: {diarization_input_tensor_for_pipeline.shape}")
        audio_for_diarization = {"waveform": diarization_input_tensor_for_pipeline, "sample_rate": target_sr}

        # --- Diarization ---
        try:
            from pyannote.audio import Pipeline
            # TODO: Add error handling for missing hf_token
            if not hf_token:
                # return (None, None, None, None, "Error: Hugging Face token is required.")
                # For now, let's create dummy output if token is missing to allow workflow testing without real diarization
                print("SpeakerDiarizerNode: Warning - Hugging Face token not provided. Skipping actual diarization.")

                # Determine the mono source waveform at original sample rate for consistent dummy output
                # This logic mirrors the one used later for actual diarization output
                final_source_waveform_mono_dummy = None
                if audio["waveform"].ndim == 3: # B, C, S
                    if audio["waveform"].shape[1] > 1:
                        final_source_waveform_mono_dummy = torch.mean(audio["waveform"][0], dim=0)
                    else:
                        final_source_waveform_mono_dummy = audio["waveform"][0].squeeze(0)
                elif audio["waveform"].ndim == 2: # B, S
                    final_source_waveform_mono_dummy = audio["waveform"][0]
                elif audio["waveform"].ndim == 1: # S
                    final_source_waveform_mono_dummy = audio["waveform"]
                else: # Should not happen if input validation is correct
                    final_source_waveform_mono_dummy = torch.zeros(input_sample_rate, device=processing_device) # Fallback

                num_dummy_speakers = 2
                dummy_summary = f"Detected {num_dummy_speakers} (dummy) speakers (HF Token missing)."
                dummy_outputs = []
                original_duration_samples_dummy = final_source_waveform_mono_dummy.shape[0]

                for i in range(num_dummy_speakers):
                    speaker_waveform = torch.zeros_like(final_source_waveform_mono_dummy)

                    mid_point = original_duration_samples_dummy // 2
                    segment_duration = min(input_sample_rate // 2, original_duration_samples_dummy // 4) # 0.5s or quarter length

                    if segment_duration > 0:
                        if i == 0: # Speaker 1 talks in first part
                            speaker_waveform[:segment_duration] = final_source_waveform_mono_dummy[:segment_duration] * 0.5
                        elif i == 1: # Speaker 2 talks in a later part
                            start_sample = mid_point
                            end_sample = mid_point + segment_duration
                            if end_sample <= original_duration_samples_dummy:
                                speaker_waveform[start_sample:end_sample] = final_source_waveform_mono_dummy[start_sample:end_sample] * 0.5
                            elif start_sample < original_duration_samples_dummy : # If segment goes out of bounds, take what's available
                                speaker_waveform[start_sample:] = final_source_waveform_mono_dummy[start_sample:original_duration_samples_dummy] * 0.5


                    dummy_outputs.append({"waveform": speaker_waveform.unsqueeze(0).unsqueeze(0), "sample_rate": input_sample_rate})

                while len(dummy_outputs) < 4:
                    silent_waveform = torch.zeros_like(final_source_waveform_mono_dummy)
                    dummy_outputs.append({"waveform": silent_waveform.unsqueeze(0).unsqueeze(0), "sample_rate": input_sample_rate})

                return tuple(dummy_outputs) + (dummy_summary,)


            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
            pipeline.to(processing_device)

            # Perform diarization
            # The input to pipeline should be a path or a dict {"waveform": tensor, "sample_rate": int}
            # The tensor should be (num_channels, num_samples) or (num_samples) for mono.
            # pyannote expects waveform on CPU, it moves to GPU internally if pipeline.to(gpu) was called.
            # diarization_waveform is currently (num_samples) on CPU.
            diarization_result = pipeline(audio_for_diarization) # audio_for_diarization is {"waveform": (1,S), "sample_rate": T_SR}

        except ImportError:
            error_message = "Error: pyannote.audio is not installed. Please install it."
            print(f"SpeakerDiarizerNode: {error_message}")
            return self.return_silent_outputs_with_message(audio, error_message)
        except Exception as e:
            import traceback
            error_message = f"Error during diarization: {str(e)}\n{traceback.format_exc()}"
            print(f"SpeakerDiarizerNode: {error_message}")
            return self.return_silent_outputs_with_message(audio, error_message)

        # --- Postprocessing and Audio Splitting ---
        print(f"SpeakerDiarizerNode: Diarization successful. Result object: {diarization_result}")
        output_audios = []

        try:
            speakers = sorted(list(diarization_result.labels()))
            print(f"SpeakerDiarizerNode: Detected speaker labels: {speakers}")
        except Exception as e:
            import traceback
            error_message = f"Error processing diarization labels: {str(e)}\n{traceback.format_exc()}"
            print(f"SpeakerDiarizerNode: {error_message}")
            return self.return_silent_outputs_with_message(audio, error_message)


        # Use original waveform for creating segments to maintain original quality and sample rate
        # input_waveform_mono was (num_samples) at original sample rate
        # diarization_waveform was (num_samples) at target_sr (16kHz)
        # input_waveform is (B, C, S) or (B,S) or (S)

        # Let's use the mono version of the input waveform at its original sample rate for segment creation.
        # If input_waveform was (B,C,S), take first batch, first channel for simplicity or average
        final_source_waveform_mono = None
        if audio["waveform"].ndim == 3: # B, C, S
            if audio["waveform"].shape[1] > 1: # Multi-channel
                final_source_waveform_mono = torch.mean(audio["waveform"][0], dim=0) # Avg channels of first batch item
            else: # Mono, but with channel dim
                final_source_waveform_mono = audio["waveform"][0].squeeze(0)
        elif audio["waveform"].ndim == 2: # B, S (assuming first item is the one)
            final_source_waveform_mono = audio["waveform"][0]
        elif audio["waveform"].ndim == 1: # S
            final_source_waveform_mono = audio["waveform"]
        else:
            return (None, None, None, None, "Error: Unexpected input audio waveform shape.")

        original_duration_samples = final_source_waveform_mono.shape[0]

        for i in range(min(len(speakers), 4)): # Max 4 output slots
            speaker_label = speakers[i]
            # Create silent audio for this speaker, with original duration and sample rate
            speaker_waveform = torch.zeros(original_duration_samples, dtype=final_source_waveform_mono.dtype, device=final_source_waveform_mono.device)

            for turn, _, label in diarization_result.itertracks(yield_label=True):
                if label == speaker_label:
                    start_sec, end_sec = turn.start, turn.end
                    start_sample = int(start_sec * input_sample_rate)
                    end_sample = int(end_sec * input_sample_rate)

                    print(f"SpeakerDiarizerNode: Processing segment for speaker {speaker_label}: start_sec={start_sec:.2f}, end_sec={end_sec:.2f}, start_sample={start_sample}, end_sample={end_sample}")

                    # Clamp to avoid going out of bounds
                    start_sample = min(max(0, start_sample), original_duration_samples)
                    end_sample = min(max(0, end_sample), original_duration_samples)

                    if end_sample > start_sample:
                        segment = final_source_waveform_mono[start_sample:end_sample]
                        speaker_waveform[start_sample:end_sample] = segment
                        print(f"SpeakerDiarizerNode: Copied segment of length {segment.shape[0]} to speaker {speaker_label}'s track.")
                    else:
                        print(f"SpeakerDiarizerNode: Skipped segment for speaker {speaker_label} as end_sample ({end_sample}) was not greater than start_sample ({start_sample}).")

            # Reshape to ComfyUI's expected (Batch, Channels, Samples)
            output_audios.append({"waveform": speaker_waveform.unsqueeze(0).unsqueeze(0), "sample_rate": input_sample_rate})
            print(f"SpeakerDiarizerNode: Created audio track for speaker {speaker_label} with total duration {speaker_waveform.shape[0]/input_sample_rate:.2f}s.")

        # Fill remaining outputs with silence if fewer than 4 speakers
        while len(output_audios) < 4:
            silent_waveform = torch.zeros(original_duration_samples, dtype=final_source_waveform_mono.dtype, device=final_source_waveform_mono.device)
            output_audios.append({"waveform": silent_waveform.unsqueeze(0).unsqueeze(0), "sample_rate": input_sample_rate})

        summary = f"Detected {len(speakers)} speakers. Outputting audio for first {min(len(speakers), 4)} speakers."
        for i, speaker_label in enumerate(speakers):
            total_time = sum(turn.duration for turn, _, lbl in diarization_result.itertracks(yield_label=True) if lbl == speaker_label)
            summary += f"\nSpeaker {speaker_label}: {total_time:.2f}s"
            if i >= 3: # Max 4 speakers in summary detail for brevity
                if len(speakers) > 4:
                    summary += "\n(and more...)"
                break

        return tuple(output_audios) + (summary,)

# Add to NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
# This will be done in multitalk/nodes.py or a main __init__.py later.
# For now, this file defines the node.
# Example of how it would be added elsewhere:
# from .diarization_nodes import SpeakerDiarizerNode
# NODE_CLASS_MAPPINGS = {
#     "SpeakerDiarizer": SpeakerDiarizerNode,
#     # ... other nodes
# }
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "SpeakerDiarizer": "Speaker Diarizer",
#     # ... other nodes
# }
