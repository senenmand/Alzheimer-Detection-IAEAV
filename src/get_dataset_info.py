import os

import pandas as pd

from src.utils import DATA_EXTRACTED_DIR, PATH_RESULTS, PATH_TRANSCRIPTIONS, apply_function_to_dataset


PATH_EMBEDDINGS = os.path.join(DATA_EXTRACTED_DIR, "embeddings.csv")
PATH_FEATURES = os.path.join(DATA_EXTRACTED_DIR, "features.csv")


def get_all_info(transcribe=False, evaluate=False, embeddings=False, features=False):
    if transcribe:
        from src.transcriptor import transcribe_audio

        df_transcriptions = apply_function_to_dataset(transcribe_audio, debug=False)
        df_transcriptions.to_csv(PATH_TRANSCRIPTIONS, index=False)
    else:
        df_transcriptions = pd.read_csv(PATH_TRANSCRIPTIONS)
        print("Loaded transcriptions from CSV.")

    if evaluate:
        from src.evaluate_transcription import EVALUATOR

        evaluator = EVALUATOR()
        df_results = evaluator.evaluate_transcriptions()
        df_results.to_csv(PATH_RESULTS, index=False)
    else:
        df_results = pd.read_csv(PATH_RESULTS)
        print("Loaded evaluation results from CSV.")

    if embeddings:
        from src.embedding_features import ExtractEmbeddings

        embedding_extractor = ExtractEmbeddings()
        df_embeddings = apply_function_to_dataset(embedding_extractor.extract_embeddings)
        df_embeddings.to_csv(PATH_EMBEDDINGS, index=False)
    else:
        df_embeddings = pd.read_csv(PATH_EMBEDDINGS)
        print("Loaded embeddings from CSV.")

    if features:
        from src.features import extract_features

        df_features = apply_function_to_dataset(extract_features)
        df_features.to_csv(PATH_FEATURES, index=False)
    else:
        df_features = pd.read_csv(PATH_FEATURES)
        print("Loaded features from CSV.")

    return df_transcriptions, df_results, df_embeddings, df_features
