import gc
import re

import numpy as np
import torch
import whisper

from src.audio_preprocessing import load_audio, parse_audio_segments
from src.config import PROMPTS_TAREA, PROMPT_GENERIC
from src.utils import group_segments_by_task


MODEL_WHISPER_NAME = "large-v3"
CLEAR_EVERY_N = 100
SEPARATION_SILENCE = np.zeros(int(0.3 * 16_000), dtype=np.float32)
MIN_SAMPLES = int(3.0 * 16_000)
SPANISH_PATTERN = re.compile(r'[^a-záéíóúüñÁÉÍÓÚÜÑ0-9\s.,;:¿?¡!\'"()\-]', re.IGNORECASE)

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = None
TRANSCRIPTIONS_DONE = 0


def clear_hooks(model):
    """Remove all registered forward hooks from a model."""
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()


def clear_gpu_memory(model=None):
    """Free GPU memory and optionally reload the Whisper model."""
    global whisper_model
    should_reload = model is not None
    if should_reload:
        try:
            model.cpu()
        except Exception:
            pass
        del model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if should_reload:
        whisper_model = whisper.load_model(MODEL_WHISPER_NAME, device=device)
        clear_hooks(whisper_model)
        print("    Reloaded Whisper model and cleared GPU memory.")
        return whisper_model
    return None


def get_whisper_model():
    """Return the shared Whisper model, loading it on demand."""
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model(MODEL_WHISPER_NAME, device=device)
        clear_hooks(whisper_model)
    return whisper_model


def get_prompt_for_task(segment_name: str) -> str:
    for task_key in PROMPTS_TAREA:
        if segment_name.upper().startswith(task_key):
            return PROMPTS_TAREA[task_key]
    return PROMPT_GENERIC


def postprocess_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(\w+)( \1){2,}", r"\1", text, flags=re.IGNORECASE)
    text = ". ".join(sentence.strip().capitalize() for sentence in text.split("."))
    return text.strip()


def is_valid_spanish_text(text: str, strange_character_threshold: float = 0.08) -> bool:
    if not text:
        return True
    strange_characters = SPANISH_PATTERN.findall(text)
    return len(strange_characters) / len(text) < strange_character_threshold


def pad_audio(audio: np.ndarray, min_samples: int = MIN_SAMPLES) -> np.ndarray:
    if len(audio) < min_samples:
        padding = np.zeros(min_samples - len(audio), dtype=np.float32)
        audio = np.concatenate([audio, padding])
    return audio


def get_task_transcription(model, full_audio, intervals, segment_name):
    fragments = []
    for start, end in intervals:
        start_sample = int(start * 16_000)
        end_sample = int(end * 16_000)
        fragments.append(full_audio[start_sample:end_sample].astype(np.float32))
        fragments.append(SEPARATION_SILENCE)

    if not fragments:
        return ""

    task_audio = np.concatenate(fragments[:-1])
    task_audio = pad_audio(task_audio)
    prompt = get_prompt_for_task(segment_name)
    anchored_prompt = "Hola, voy a hablar en español. " + prompt

    for temperature in [0.0, 0.2, 0.4]:
        result = model.transcribe(
            task_audio,
            language="es",
            task="transcribe",
            fp16=(device == "cuda"),
            beam_size=5 if temperature == 0.0 else 1,
            temperature=temperature,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            initial_prompt=anchored_prompt,
        )
        text = " ".join(segment["text"] for segment in result["segments"])
        text = postprocess_text(text)
        if is_valid_spanish_text(text):
            if temperature > 0.0:
                print(f"      Retry with temperature={temperature} succeeded.")
            return text
        print(f"      Hallucination detected (temp={temperature}): {text[:60]}…")

    print(f"      Could not obtain a valid Spanish transcription for {segment_name}.")
    return ""


def transcribe_audio(path_audio, path_label):
    global TRANSCRIPTIONS_DONE, whisper_model

    model = get_whisper_model()
    audio = load_audio(path_audio)
    segments = parse_audio_segments(path_label)
    groups = group_segments_by_task(segments)

    transcriptions = {
        task: get_task_transcription(model, audio, intervals, task)
        for task, intervals in groups.items()
    }

    TRANSCRIPTIONS_DONE += 1
    if TRANSCRIPTIONS_DONE % CLEAR_EVERY_N == 0:
        whisper_model = clear_gpu_memory(model)

    return transcriptions
