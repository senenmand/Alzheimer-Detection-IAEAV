import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR", "")

PATH_DIR_AUDIO_CONTROLS = os.path.join(DATA_DIR, "audio", "controls")
PATH_DIR_AUDIO_PATIENTS = os.path.join(DATA_DIR, "audio", "patients")

PATH_DIR_LABELS_CONTROLS = os.path.join(DATA_DIR, "labels", "controls")
PATH_DIR_LABELS_PATIENTS = os.path.join(DATA_DIR, "labels", "patients")

PATH_METADATA = os.path.join(DATA_DIR, "metadata.csv")

DATA_EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")
os.makedirs(DATA_EXTRACTED_DIR, exist_ok=True)

PATH_TRANSCRIPTIONS = os.path.join(DATA_EXTRACTED_DIR, "transcriptions.csv")
PATH_RESULTS = os.path.join(DATA_EXTRACTED_DIR, "final_results.csv")


def get_patient_id(filename):
    """Return the participant ID from a file name without its extension."""
    return os.path.splitext(filename)[0]


def group_segments_by_task(segments):
    """Group task segments while preserving their original order."""
    groups = {}
    for start, end, name in segments:
        groups.setdefault(name, []).append((start, end))
    return groups


def apply_function_to_dataset(function, debug=False):
    audio_paths = sorted(os.listdir(PATH_DIR_AUDIO_CONTROLS)) + sorted(os.listdir(PATH_DIR_AUDIO_PATIENTS))
    label_paths = sorted(os.listdir(PATH_DIR_LABELS_CONTROLS)) + sorted(os.listdir(PATH_DIR_LABELS_PATIENTS))

    participant_ids = [get_patient_id(filename) for filename in audio_paths]
    labels = [0] * len(os.listdir(PATH_DIR_AUDIO_CONTROLS)) + [1] * len(os.listdir(PATH_DIR_AUDIO_PATIENTS))

    dataset = pd.DataFrame({"id": participant_ids, "label": labels})
    rows = []

    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        patient_id = row["id"]
        label = row["label"]

        audio_filename = audio_paths[index]
        label_filename = label_paths[index]

        audio_path = os.path.join(
            PATH_DIR_AUDIO_CONTROLS if label == 0 else PATH_DIR_AUDIO_PATIENTS,
            audio_filename,
        )
        label_path = os.path.join(
            PATH_DIR_LABELS_CONTROLS if label == 0 else PATH_DIR_LABELS_PATIENTS,
            label_filename,
        )

        result = function(audio_path, label_path)

        if debug:
            print(audio_path, label_path, result)

        flattened_row = {"id": patient_id, "label": label}
        for key, value in result.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    flattened_row[nested_key] = nested_value
            else:
                flattened_row[key] = value
        rows.append(flattened_row)

    return pd.DataFrame(rows)


def create_5cv(dict_id_labels, val=False, n_splits=5, random_state=42):
    """Create rotating 5-fold splits at participant level for bimodal data."""
    ids = np.array(list(dict_id_labels.keys()))
    unique_ids = np.unique(ids)
    patient_labels = np.array([dict_id_labels[patient_id] for patient_id in unique_ids])

    stratified_kfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    patient_folds = [unique_ids[indexes] for _, indexes in stratified_kfold.split(unique_ids, patient_labels)]

    folds = []
    for fold_index in range(n_splits):
        test_ids = patient_folds[fold_index]

        if val:
            val_ids = patient_folds[(fold_index + 1) % n_splits]
            train_ids = np.concatenate(
                [
                    patient_folds[index]
                    for index in range(n_splits)
                    if index not in [fold_index, (fold_index + 1) % n_splits]
                ]
            )
        else:
            val_ids = None
            train_ids = np.concatenate(
                [patient_folds[index] for index in range(n_splits) if index != fold_index]
            )

        folds.append({"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids})

    return folds
