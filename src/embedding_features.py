import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor

from src.utils import PATH_TRANSCRIPTIONS, get_patient_id

SR = 16_000

class ExtractEmbeddings:
    def __init__(self):
        self.SR = SR
        self.SILENCE_SEP = np.zeros(int(0.5 * self.SR), dtype=np.float32)
        self.MIN_AUDIO_DURATION = int(1.0 * self.SR)

        self.df_transcriptions = pd.read_csv(PATH_TRANSCRIPTIONS).set_index("id")

        self.model_name_text = "dccuchile/bert-base-spanish-wwm-cased"
        self.model_name_audio = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer_text = AutoTokenizer.from_pretrained(self.model_name_text)
        self.model_text = AutoModel.from_pretrained(self.model_name_text).to(self.device)

        self.processor_audio = Wav2Vec2Processor.from_pretrained(self.model_name_audio)
        self.model_audio = Wav2Vec2Model.from_pretrained(self.model_name_audio).to(self.device)

    def load_audio(self, path: str, target_sr: int = 16_000) -> np.ndarray:
        "Load an audio file, convert to mono, and resample to the target sampling rate if necessary."
        waveform, original_sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform.squeeze().numpy().astype(np.float32)

    def parse_audio_segments(self, txt_file):
        "Parse the audio segments from the given text file and return a list of (start, end, task) tuples."
        labels_df = pd.read_csv(txt_file, sep="	", header=None)
        labels_df[3] = labels_df[2].where(labels_df[2].str.startswith("T")).ffill()
        labels_df = labels_df[labels_df[2] == "PAC"]
        return labels_df[[0, 1, 3]].values.tolist()

    def group_segments_by_task(self, segments):
        "Group the audio segments by their task name and return a dictionary where keys are task names and values are lists of (start, end) tuples."
        groups = {}
        for start, end, name in segments:
            groups.setdefault(name, []).append((start, end))
        return groups

    def extract_embeddings_text(self, texts):
        "Extract text embeddings for the given texts."
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer_text(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model_text(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

    def concat_audio(self, full_audio, intervals):
        "Concatenate audio segments based on the provided intervals, inserting silence between segments. If no segments are provided, return a silent audio of minimum duration."
        fragments = []
        for _, segments in intervals.items():
            for start, end in segments:
                start_sample = int(start * self.SR)
                end_sample = int(end * self.SR)
                fragments.append(full_audio[start_sample:end_sample].astype(np.float32))
                fragments.append(self.SILENCE_SEP)
        if len(fragments) > 1:
            audio = np.concatenate(fragments[:-1])
        elif len(fragments) == 1:
            audio = fragments[0]
        else:
            audio = np.zeros(self.MIN_AUDIO_DURATION, dtype=np.float32)
        if len(audio) < self.MIN_AUDIO_DURATION:
            audio = np.concatenate([audio, np.zeros(self.MIN_AUDIO_DURATION - len(audio), dtype=np.float32)])
        return audio

    def extract_embeddings_audio(self, audios, sampling_rate=16_000):
        "Extract audio embeddings for the given audio samples."
        if isinstance(audios, np.ndarray):
            audios = [audios]
        inputs = self.processor_audio(audios, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
        with torch.no_grad():
            outputs = self.model_audio(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

    def extract_embeddings(self, path_audio, path_label, sampling_rate=16_000):
        "Extract both audio and text embeddings for the given audio file and its corresponding label file."
        audio = self.load_audio(path_audio)
        segments = self.parse_audio_segments(path_label)
        task_groups = self.group_segments_by_task(segments)
        audio = self.concat_audio(audio, task_groups)

        patient_id = get_patient_id(os.path.basename(path_audio))
        row = self.df_transcriptions.loc[patient_id]
        text = ". ".join(
            str(row[column]) for column in self.df_transcriptions.columns if column.startswith("T")
        ) + "."

        audio_embedding = self.extract_embeddings_audio(audio, sampling_rate)[0]
        text_embedding = self.extract_embeddings_text(text)[0]

        embedding_result = {
            f"embedding_audio_{index}": float(value)
            for index, value in enumerate(audio_embedding)
        }
        embedding_result.update(
            {
                f"embedding_text_{index}": float(value)
                for index, value in enumerate(text_embedding)
            }
        )
        return embedding_result
