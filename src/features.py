import os
import re
from collections import Counter
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import spacy

from src.utils import PATH_TRANSCRIPTIONS, get_patient_id
from src.audio_preprocessing import load_audio,concat_audio, parse_audio_segments, get_tasks_from_segments


def audio_features_from_waveform(
    y,
    sr,
    hop_length=512,
    min_pause_dur=0.3,
    merge_pause_gap=0.1,
    smooth_window_ms=50,
    syllable_env_hop=256,
    syllable_min_separation_s=0.10,
    syllable_prominence=0.1,
    n_mfcc=5,
    mfcc_max_k=10,
    n_fft=1024,
):
    """Extract scalar acoustic features from a waveform."""
    try:
        import parselmouth
    except Exception:
        parselmouth = None

    y = np.asarray(y, dtype=np.float32).squeeze()
    if y.ndim != 1:
        raise ValueError(f"Waveform must be 1D. Received shape={y.shape}")
    if y.size == 0:
        y = np.zeros(int(sr * 1.0), dtype=np.float32)

    total_duration_s = len(y) / sr
    total_minutes = total_duration_s / 60 if total_duration_s > 0 else 1e-8

    rms = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=n_fft)[0]
    frame_duration = hop_length / sr
    n_frames = len(rms)

    rms_threshold = np.percentile(rms, 15)
    silence_raw = rms < rms_threshold

    window_size = max(1, int((smooth_window_ms / 1000) / frame_duration))
    kernel = np.ones(window_size) / window_size
    silence_smooth = np.convolve(silence_raw.astype(float), kernel, mode="same") > 0.5

    pauses = []
    in_pause, pause_start = False, 0
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
        else:
            previous = merged_pauses[-1]
            gap = (pause[0] - previous[1]) * frame_duration
            if gap < merge_pause_gap:
                merged_pauses[-1] = (previous[0], pause[1], (pause[1] - previous[0]) * frame_duration)
            else:
                merged_pauses.append(pause)

    pause_durations = np.array([pause[2] for pause in merged_pauses], dtype=float)
    total_pauses = int(pause_durations.size)
    total_silence_s = float(np.sum(pause_durations)) if total_pauses else 0.0

    silence_percentage = float(100 * total_silence_s / total_duration_s) if total_duration_s > 0 else 0.0
    pause_ratio = float(total_silence_s / total_duration_s) if total_duration_s > 0 else 0.0
    pauses_per_minute = float(total_pauses / total_minutes)

    speech = ~silence_smooth
    speech_segments = []
    in_segment, segment_start = False, 0
    for frame_index, is_speech in enumerate(speech):
        if is_speech and not in_segment:
            in_segment, segment_start = True, frame_index
        elif not is_speech and in_segment:
            speech_segments.append((segment_start, frame_index))
            in_segment = False
    if in_segment:
        speech_segments.append((segment_start, n_frames))

    speech_durations = np.array([(end - start) * frame_duration for start, end in speech_segments], dtype=float)
    num_speech_segments = int(speech_durations.size)
    speech_fraction = float(np.mean(speech)) if n_frames else 0.0
    mean_speech_segment_dur = float(np.mean(speech_durations)) if num_speech_segments else 0.0
    speech_segment_p90 = float(np.percentile(speech_durations, 90)) if num_speech_segments else 0.0

    f0_mean = 0.0
    f0_std = 0.0
    hnr_mean = 0.0
    if parselmouth is not None:
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        pitch = sound.to_pitch(time_step=hop_length / sr, pitch_floor=50, pitch_ceiling=500)
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        f0_mean = float(np.mean(f0)) if len(f0) else 0.0
        f0_std = float(np.std(f0)) if len(f0) else 0.0
        try:
            harmonicity = sound.to_harmonicity()
            harmonicity_values = harmonicity.values
            harmonicity_values = harmonicity_values[harmonicity_values > -200]
            hnr_mean = float(np.mean(harmonicity_values)) if len(harmonicity_values) else 0.0
        except Exception:
            hnr_mean = 0.0

    envelope = librosa.feature.rms(y=y, hop_length=syllable_env_hop, frame_length=n_fft)[0]
    if envelope.size >= 3:
        envelope_normalized = (envelope - np.min(envelope)) / (np.ptp(envelope) + 1e-12)
        peaks = []
        for index in range(1, len(envelope_normalized) - 1):
            if (
                envelope_normalized[index] > envelope_normalized[index - 1]
                and envelope_normalized[index] > envelope_normalized[index + 1]
                and envelope_normalized[index] >= syllable_prominence
            ):
                peaks.append(index)

        min_sep_frames = max(1, int(syllable_min_separation_s * sr / syllable_env_hop))
        filtered_peaks = []
        last_peak = -10**9
        for peak in peaks:
            if peak - last_peak >= min_sep_frames:
                filtered_peaks.append(peak)
                last_peak = peak

        syllable_count = int(len(filtered_peaks))
        syllable_rate = float(syllable_count / total_duration_s) if total_duration_s > 0 else 0.0
    else:
        syllable_count = 0
        syllable_rate = 0.0

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    max_k = min(mfcc_max_k, mfcc.shape[0] - 1)
    mfcc_features = {}
    for k in range(0, max_k + 1):
        mfcc_features[f"waveform_mfcc_{k}_mean"] = float(np.mean(mfcc[k]))
        mfcc_features[f"waveform_mfcc_{k}_var"] = float(np.var(mfcc[k]))

    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    rms_std = float(np.std(rms)) if rms.size else 0.0

    return {
        "waveform_f0_mean": float(f0_mean),
        "waveform_f0_std": float(f0_std),
        "waveform_hnr_mean": float(hnr_mean),
        "waveform_syllable_count": int(syllable_count),
        "waveform_syllable_rate": float(syllable_rate),
        "waveform_rms_mean": rms_mean,
        "waveform_rms_std": rms_std,
        "waveform_rms_threshold": float(rms_threshold),
        "pauses_per_minute": pauses_per_minute,
        "pause_ratio": pause_ratio,
        "silence_percentage": silence_percentage,
        "num_speech_segments": num_speech_segments,
        "mean_speech_segment_dur": mean_speech_segment_dur,
        "speech_segment_p90": speech_segment_p90,
        "speech_fraction": speech_fraction,
        **mfcc_features,
    }


def analyze_mel(
    logmel,
    sr,
    hop_length=512,
    min_pause_dur=0.3,
    merge_pause_gap=0.1,
    energy_percentile=10,
    smooth_window_ms=50,
    debug=False,
):
    """Extract log-Mel features related to fluency and speech fragmentation."""
    if not isinstance(logmel, np.ndarray):
        logmel = logmel.detach().cpu().numpy()
    logmel = np.squeeze(logmel)
    if logmel.ndim != 2:
        raise ValueError(f"logmel must be (n_mels, n_frames), received {logmel.shape}")

    frame_duration = hop_length / sr
    _, n_frames = logmel.shape
    total_audio_duration = n_frames * frame_duration
    total_minutes = total_audio_duration / 60 if total_audio_duration > 0 else 1e-8

    frame_energy = logmel.mean(axis=0)
    energy_mean = float(np.mean(frame_energy))
    energy_std = float(np.std(frame_energy))
    energy_threshold = float(np.percentile(frame_energy, energy_percentile))

    silence_raw = frame_energy < energy_threshold
    window_size = max(1, int((smooth_window_ms / 1000) / frame_duration))
    kernel = np.ones(window_size) / window_size
    silence_smooth = np.convolve(silence_raw.astype(float), kernel, mode="same") > 0.5

    speaking_energy_mean = float(np.mean(frame_energy[~silence_smooth])) if np.any(~silence_smooth) else energy_mean
    silence_energy_mean = float(np.mean(frame_energy[silence_smooth])) if np.any(silence_smooth) else energy_mean

    pauses = []
    in_pause, pause_start = False, 0
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
        else:
            previous = merged_pauses[-1]
            gap = (pause[0] - previous[1]) * frame_duration
            if gap < merge_pause_gap:
                merged_pauses[-1] = (previous[0], pause[1], (pause[1] - previous[0]) * frame_duration)
            else:
                merged_pauses.append(pause)

    pause_durations = np.array([pause[2] for pause in merged_pauses], dtype=float)
    total_pauses = int(pause_durations.size)
    total_silence_duration = float(np.sum(pause_durations)) if total_pauses else 0.0
    silence_percentage = float(100 * total_silence_duration / total_audio_duration) if total_audio_duration > 0 else 0.0
    pause_ratio = float(total_silence_duration / total_audio_duration) if total_audio_duration > 0 else 0.0
    pause_mean = float(np.mean(pause_durations)) if total_pauses else 0.0
    pause_max = float(np.max(pause_durations)) if total_pauses else 0.0
    pauses_per_minute = float(total_pauses / total_minutes) if total_minutes > 0 else 0.0

    speech = ~silence_smooth
    speech_segments = []
    in_segment, segment_start = False, 0
    for frame_index, is_speech in enumerate(speech):
        if is_speech and not in_segment:
            in_segment, segment_start = True, frame_index
        elif not is_speech and in_segment:
            speech_segments.append((segment_start, frame_index))
            in_segment = False
    if in_segment:
        speech_segments.append((segment_start, n_frames))

    speech_durations = np.array([(end - start) * frame_duration for start, end in speech_segments], dtype=float)
    num_speech_segments = int(speech_durations.size)
    speech_fraction = float(np.mean(speech))
    mean_speech_segment_dur = float(np.mean(speech_durations)) if num_speech_segments else 0.0
    speech_segment_p90 = float(np.percentile(speech_durations, 90)) if num_speech_segments else 0.0
    speech_frames = int(np.sum(speech))
    speech_rate = float(speech_frames / total_minutes)
    energy_delta_std = float(np.std(np.diff(frame_energy, prepend=frame_energy[0])))

    return {
        "pauses_per_minute": pauses_per_minute,
        "pause_ratio": pause_ratio,
        "silence_percentage": silence_percentage,
        "pause_mean": pause_mean,
        "pause_max": pause_max,
        "num_speech_segments": num_speech_segments,
        "mean_speech_segment_dur": mean_speech_segment_dur,
        "speech_segment_p90": speech_segment_p90,
        "speech_fraction": speech_fraction,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "speaking_energy_mean": speaking_energy_mean,
        "silence_energy_mean": silence_energy_mean,
        "energy_delta_std": energy_delta_std,
        "speech_rate": speech_rate,
        "energy_threshold": float(energy_threshold),
    }


DISFLUENCY_FILLERS = {"eh", "um", "mm", "mmm", "ah", "uh", "em", "eeh", "amm", "hmm", "am"}
DISCOURSE_FILLERS = {
    "bueno", "osea", "pues", "este", "entonces", "vamos",
    "venga", "mira", "oye", "hombre", "claro", "vale", "buenas",
}
DISCOURSE_MULTIWORD_FILLERS = {
    "o sea", "es decir", "o sea que", "en plan", "o sea pues",
    "a ver", "es que", "la verdad", "o algo así",
}
VAGUE_WORDS = {
    "cosa", "cosas", "algo", "algunos", "muchos", "bastante",
    "varios", "tipo", "así", "tal", "tales", "eso", "esto",
}
NEGATIVE_WORDS = {
    "miedo", "temor", "aterrorizado", "aterrorizada", "aterrado", "asustado", "asustada",
    "pánico", "nervioso", "nerviosa", "triste", "ansioso", "ansiosa", "llora",
    "preocupado", "preocupada", "depresivo", "depresiva", "inquieto", "inquieta", "mal", "infeliz",
    "susto", "inseguro", "insegura", "desconcertado", "desconcertada", "abatido", "abatida",
    "no", "nunca", "jamás", "sin", "falta", "nada", "nadie", "ningún", "ninguna",
}


def linguistic_features(text: str, nlp=None, duration_min: Optional[float] = None) -> dict:
    if nlp is None:
        nlp = spacy.load("es_core_news_md")

    doc = nlp(text)
    tokens = [token for token in doc if not token.is_space]
    words = [token.text.lower() for token in tokens if token.is_alpha]
    sentences = list(doc.sents)
    n_words = max(len(words), 1)

    n_disfluency = sum(1 for word in words if word in DISFLUENCY_FILLERS)
    n_discourse = sum(1 for word in words if word in DISCOURSE_FILLERS)
    n_vague = sum(1 for word in words if word in VAGUE_WORDS)
    text_lower = text.lower()
    for marker in DISCOURSE_MULTIWORD_FILLERS:
        n_discourse += text_lower.count(marker)

    total_fillers = n_disfluency + n_discourse
    gap_fillers_per_min = (total_fillers / duration_min) if duration_min else np.nan

    sentence_lengths = np.array([len([token for token in sentence if token.is_alpha]) for sentence in sentences], dtype=float)

    def safe_stat(function, array):
        return float(function(array)) if len(array) else np.nan

    pos_counts = Counter(token.pos_ for token in tokens)
    n_nouns = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    n_verbs = pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)
    n_pronouns = pos_counts.get("PRON", 0)

    n_negative_particles = sum(1 for token in tokens if token.dep_ == "neg" or token.lemma_.lower() == "no")
    n_negative_lexical = sum(1 for token in tokens if token.lemma_.lower() in NEGATIVE_WORDS)
    n_negative = n_negative_particles + n_negative_lexical

    word_frequency = Counter(words)
    type_token_ratio = len(word_frequency) / n_words
    hapax_ratio = sum(1 for count in word_frequency.values() if count == 1) / n_words
    repetitions = sum(count - 1 for count in word_frequency.values() if count > 1)

    content_pos = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
    n_content = sum(1 for token in tokens if token.pos_ in content_pos)
    content_density = n_content / n_words

    word_lengths = np.array([len(word) for word in words], dtype=float)

    bigrams = list(zip(words[:-1], words[1:])) if len(words) >= 2 else []
    trigrams = list(zip(words[:-2], words[1:-1], words[2:])) if len(words) >= 3 else []
    bigram_frequency = Counter(bigrams)
    trigram_frequency = Counter(trigrams)

    bigram_unique_norm = (len(bigram_frequency) / n_words) if n_words else 0.0
    trigram_unique_norm = (len(trigram_frequency) / n_words) if n_words else 0.0
    bigram_repeated_norm = (sum(count - 1 for count in bigram_frequency.values() if count > 1) / n_words) if n_words else 0.0
    trigram_repeated_norm = (sum(count - 1 for count in trigram_frequency.values() if count > 1) / n_words) if n_words else 0.0

    return {
        "gap_fillers_per_min": gap_fillers_per_min,
        "ttr": type_token_ratio,
        "vocab_size": len(word_frequency),
        "hapax_ratio": hapax_ratio,
        "mean_word_len": safe_stat(np.mean, word_lengths),
        "repetitions": repetitions,
        "content_density": content_density,
        "noun_ratio": n_nouns / n_words,
        "verb_ratio": n_verbs / n_words,
        "pronoun_ratio": n_pronouns / n_words,
        "neg_ratio": n_negative / n_words,
        "mls": safe_stat(np.mean, sentence_lengths),
        "mls_std": safe_stat(np.std, sentence_lengths),
        "speech_rate": (n_words / duration_min) if duration_min else np.nan,
        "bigram_unique_norm": float(bigram_unique_norm),
        "trigram_unique_norm": float(trigram_unique_norm),
        "bigram_repeated_norm": float(bigram_repeated_norm),
        "trigram_repeated_norm": float(trigram_repeated_norm),
        "vague_ratio": n_vague / n_words,
    }


def extract_features(path_audio, path_label, sampling_rate=16_000, top_db=20):
    """Extract acoustic and linguistic features for one participant."""
    df_transcriptions = pd.read_csv(PATH_TRANSCRIPTIONS).set_index("id")

    waveform = load_audio(path_audio, target_sr=sampling_rate)
    segments = parse_audio_segments(path_label)
    task_groups = get_tasks_from_segments(segments)
    relevant_audio = concat_audio(waveform, task_groups, sr=sampling_rate)
    mel = librosa.feature.melspectrogram(
        y=relevant_audio,
        sr=sampling_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    patient_id = get_patient_id(os.path.basename(path_audio))
    row = df_transcriptions.loc[patient_id]
    text = ". ".join(str(row[column]) for column in df_transcriptions.columns if column.startswith("T")) + "."

    waveform_features = audio_features_from_waveform(relevant_audio, sampling_rate)
    mel_features = analyze_mel(logmel, sampling_rate)
    language_features = linguistic_features(text, duration_min=len(relevant_audio) / sampling_rate)

    features = {}
    features.update(waveform_features)
    features.update(mel_features)
    features.update(language_features)
    return features
