import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T


SR_DEFAULT = 16_000


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert audio to mono by averaging channels when needed."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def load_audio(path: str, target_sr: int = SR_DEFAULT) -> np.ndarray:
    """Load an audio file, convert it to mono, and resample it."""
    waveform, original_sr = torchaudio.load(path)
    waveform = to_mono(waveform)
    if original_sr != target_sr:
        resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy().astype(np.float32)


def get_logmel_file(path, n_mels=128, n_fft=1024, hop_length=512):
    """Compute the log-Mel spectrogram of an audio file."""
    waveform, sample_rate = torchaudio.load(path)
    waveform = to_mono(waveform)
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )(waveform)
    log_mel_spectrogram = T.AmplitudeToDB()(mel_spectrogram)
    return log_mel_spectrogram, sample_rate


def parse_audio_segments(txt_file):
    """Parse a TSV label file and return patient speech segments."""
    labels_df = pd.read_csv(txt_file, sep="	", header=None)
    labels_df[3] = labels_df[2].where(labels_df[2].str.startswith("T")).ffill()
    labels_df = labels_df[labels_df[2] == "PAC"]
    return labels_df[[0, 1, 3]].values.tolist()


def get_tasks_from_segments(segments):
    """Convert a list of segments into a task-to-intervals dictionary."""
    tasks = {}
    for start, end, task in segments:
        tasks.setdefault(task, []).append((start, end))
    return tasks


def concat_audio(full_audio, intervals, sr=SR_DEFAULT):
    """Concatenate audio fragments and insert short silence gaps."""
    fragments = []
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    min_duration = int(1.0 * sr)

    for start, end in intervals:
        start_sample, end_sample = int(start * sr), int(end * sr)
        fragments.append(full_audio[start_sample:end_sample].astype(np.float32))
        fragments.append(silence)

    if len(fragments) <= 1:
        audio = np.zeros(min_duration, dtype=np.float32)
    else:
        audio = np.concatenate(fragments[:-1])
        if len(audio) < min_duration:
            audio = np.concatenate([audio, np.zeros(min_duration - len(audio), dtype=np.float32)])
    return audio


def logmel_patients(audio_path, segment_groups, n_mels=128, n_fft=1024, hop_length=512):
    """Concatenate relevant segments and compute a normalized log-Mel spectrogram."""
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = to_mono(waveform)
    total_samples = waveform.shape[1]

    segments = []
    for task_segments in segment_groups.values():
        for start, end in task_segments:
            start_sample = max(0, int(start * sample_rate))
            end_sample = min(int(end * sample_rate), total_samples)
            if end_sample > start_sample:
                segments.append(waveform[:, start_sample:end_sample])

    if not segments:
        return None, sample_rate

    concatenated_waveform = torch.cat(segments, dim=1)
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )(concatenated_waveform).clamp(min=1e-10)
    log_mel_spectrogram = 10 * torch.log10(mel_spectrogram)
    log_mel_spectrogram -= log_mel_spectrogram.max()
    return log_mel_spectrogram, sample_rate


def fragment_audio(audio, sr=16_000, win_length_sec=2, hop_length_sec=1.5):
    """Split an audio array into overlapping fragments."""
    segment_samples = int(sr * win_length_sec)
    hop_samples = int(sr * hop_length_sec)
    segments = []

    for start in range(0, len(audio) - segment_samples + 1, hop_samples):
        end = start + segment_samples
        segments.append(audio[start:end])

    return segments


def create_logmel_transforms(sr=16_000, n_fft=1024, hop_length=512, n_mels=128):
    """Build the torchaudio transforms required for log-Mel extraction."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()
    return mel_transform, db_transform


def get_logmel(segment, mel_transform, db_transform):
    """Convert one audio segment into a log-Mel spectrogram."""
    segment_tensor = torch.from_numpy(segment).unsqueeze(0)
    mel = mel_transform(segment_tensor)
    logmel = db_transform(mel)
    return logmel.squeeze(0)


def get_logmels(segments, mel_transform, db_transform):
    """Convert a list of audio segments into log-Mel spectrograms."""
    return [get_logmel(segment, mel_transform, db_transform) for segment in segments]


def fragment_and_logmel(
    audio,
    sr=16_000,
    win_length_sec=2,
    hop_length_sec=1.5,
    n_mels=128,
    n_fft=1024,
    hop_length=512,
):
    """Split audio into fragments and compute a log-Mel spectrogram for each one."""
    segments = fragment_audio(
        audio,
        sr=sr,
        win_length_sec=win_length_sec,
        hop_length_sec=hop_length_sec,
    )
    mel_transform, db_transform = create_logmel_transforms(
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    return get_logmels(segments, mel_transform, db_transform)


def analyze_mel(
    logmel,
    sr,
    hop_length=512,
    min_pause_dur=0.3,
    merge_pause_gap=0.1,
    energy_percentile=5,
    smooth_window_ms=50,
    debug=False,
):
    """Extract temporal, energy, and silence metrics from a log-Mel spectrogram."""
    if not isinstance(logmel, np.ndarray):
        logmel = logmel.detach().cpu().numpy()
    logmel = np.squeeze(logmel)
    if logmel.ndim != 2:
        raise ValueError(f"logmel must be (n_mels, n_frames), received {logmel.shape}")

    frame_duration = hop_length / sr
    n_frames = logmel.shape[1]
    total_audio_duration = n_frames * frame_duration
    total_minutes = total_audio_duration / 60 if total_audio_duration > 0 else 1e-8

    frame_energy = logmel.mean(axis=0)
    energy_mean = float(frame_energy.mean())
    energy_std = float(frame_energy.std())
    energy_min = float(frame_energy.min())
    energy_max = float(frame_energy.max())

    spectral_flux = float(np.mean(np.abs(np.diff(logmel, axis=1))))
    mel_norm = logmel / (np.sum(logmel, axis=0, keepdims=True) + 1e-8)
    spectral_entropy = float(-np.sum(mel_norm * np.log(mel_norm + 1e-8), axis=0).mean())

    threshold_percentile = max(1, int(len(frame_energy) * energy_percentile / 100))
    low_energy_frames = np.sort(frame_energy)[:threshold_percentile]
    noise_floor = np.median(low_energy_frames)
    energy_threshold = noise_floor + 3.0

    silence_raw = frame_energy < energy_threshold
    window_size = max(1, int((smooth_window_ms / 1000) / frame_duration))
    kernel = np.ones(window_size) / window_size
    silence_smooth = np.convolve(silence_raw.astype(float), kernel, mode="same") > 0.5

    pauses, in_pause, pause_start = [], False, 0
    for frame_index, is_silent in enumerate(silence_smooth):
        if is_silent and not in_pause:
            in_pause, pause_start = True, frame_index
        elif not is_silent and in_pause:
            pauses.append((pause_start, frame_index))
            in_pause = False
    if in_pause:
        pauses.append((pause_start, n_frames))

    pauses_seconds = [
        (start, end, (end - start) * frame_duration)
        for start, end in pauses
        if (end - start) * frame_duration >= min_pause_dur
    ]

    merged_pauses = []
    for pause in pauses_seconds:
        if not merged_pauses:
            merged_pauses.append(pause)
            continue
        previous = merged_pauses[-1]
        gap = (pause[0] - previous[1]) * frame_duration
        if gap < merge_pause_gap:
            merged_pauses[-1] = (previous[0], pause[1], (pause[1] - previous[0]) * frame_duration)
        else:
            merged_pauses.append(pause)

    pause_durations = [pause[2] for pause in merged_pauses]
    total_pauses = len(pause_durations)
    total_silence_duration = sum(pause_durations) if pause_durations else 0.0
    silence_percentage = (
        100 * total_silence_duration / total_audio_duration if total_audio_duration > 0 else 0.0
    )
    mean_pause_duration = float(np.mean(pause_durations)) if pause_durations else 0.0

    silence_frames = int(np.sum(silence_smooth))
    speech_frames = n_frames - silence_frames
    speech_rate = speech_frames / total_minutes
    pause_ratio = total_silence_duration / total_audio_duration if total_audio_duration > 0 else 0.0

    return {
        "long_pauses_per_minute": len([duration for duration in pause_durations if duration >= 1]) / total_minutes,
        "mean_pause_duration": mean_pause_duration,
        "energy_std": energy_std,
        "spectral_flux": spectral_flux,
        "spectral_entropy": spectral_entropy,
        "pauses_per_minute": total_pauses / total_minutes,
        "silence_percentage": silence_percentage,
        "energy_mean": energy_mean,
        "energy_min": energy_min,
        "energy_max": energy_max,
        "total_pauses": total_pauses,
        "pause_durations": pause_durations,
        "energy_threshold": float(energy_threshold),
        "noise_floor": float(noise_floor),
        "speech_rate": float(speech_rate),
        "pause_ratio": float(pause_ratio),
    }
