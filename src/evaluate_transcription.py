import os
import re
import unicodedata
from datetime import date

import pandas as pd
from names_dataset import NameDataset
from rapidfuzz import fuzz, process

from src.utils import (
    PATH_DIR_LABELS_CONTROLS,
    PATH_DIR_LABELS_PATIENTS,
    PATH_METADATA,
    PATH_TRANSCRIPTIONS,
)


def parse_audio_segments(txt_file):
    """Parse a TSV label file and return its annotated segments."""
    labels_df = pd.read_csv(txt_file, sep="	", header=None)
    labels_df[3] = labels_df[2].where(labels_df[2].str.startswith("T")).ffill()
    return labels_df


class EVALUATOR:
    def __init__(self):
        self.MESES = [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
        ]
        self.MONTHS_INDEX = {month: index + 1 for index, month in enumerate(self.MESES)}

        self.DIAS_SEMANA = {
            "lunes": 0,
            "martes": 1,
            "miercoles": 2,
            "miércoles": 2,
            "jueves": 3,
            "viernes": 4,
            "sabado": 5,
            "sábado": 5,
            "domingo": 6,
        }

        self.UNIDADES = {
            "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
            "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
            "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
            "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
            "veinte": 20, "veintiuno": 21, "veintidos": 22, "veintitres": 23,
            "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
            "veintisiete": 27, "veintiocho": 28, "veintinueve": 29, "treinta": 30,
        }
        self.CENTENAS = {
            "cien": 100, "ciento": 100, "doscientos": 200, "trescientos": 300,
            "cuatrocientos": 400, "quinientos": 500, "seiscientos": 600,
            "setecientos": 700, "ochocientos": 800, "novecientos": 900,
        }
        self.NUMBERS_ES = {**self.UNIDADES, **self.CENTENAS}

        self.WORDS_TO_EXCLUDE = {
            "de", "la", "el", "los", "las", "un", "una", "unos", "unas",
            "y", "o", "no", "se", "me", "te", "le", "lo", "ya", "si",
            "que", "en", "con", "por", "para", "del", "al", "hay",
            "mas", "pero", "como", "pues", "ahora", "sé", "creo",
        }
        self.names_dataset = NameDataset()

        self.OBJECTS_LIST = ["Baraja", "Coche", "Pera", "Zapato", "Cuchara", "Trompeta"]
        self.SYNONYMS = {
            "Baraja": ["baraja", "cartas", "carta", "naipes", "laipes", "maraja"],
            "Coche": ["coche", "vehiculo", "carro", "auto"],
            "Pera": ["pera", "peras", "pira", "pedra", "vela"],
            "Zapato": ["zapato", "zapatos"],
            "Cuchara": ["cuchara", "cucharas", "luchara"],
            "Trompeta": ["trompeta", "trompetas", "corneta"],
        }
        self.VOCABULARY = []
        self.VARIANT_MAP = {}
        for obj, synonyms in self.SYNONYMS.items():
            for synonym in synonyms:
                self.VOCABULARY.append(synonym)
                self.VARIANT_MAP[synonym] = obj

        self.ELEMENTS_LIST = ["Niño", "Madre", "Coche", "Pelota", "Carretera", "Peligro"]
        self.ELEMENTS_SYNONYMS = {
            "Niño": ["Niño", "niño", "niña", "chico", "chica", "muchacho", "muchacha", "nene"],
            "Madre": ["Madre", "madre", "mamá", "mujer", "señora"],
            "Coche": ["Coche", "coche", "auto", "vehículo", "carro"],
            "Pelota": ["Pelota", "pelota", "balón", "jugar"],
            "Carretera": ["Carretera", "carretera", "calle", "asfalto", "en la carretera", "en la calle", "cruzando", "calzada"],
            "Peligro": ["Peligro", "peligro", "accidente", "atropellar", "va a pasar", "riesgo", "asustada", "asustado", "gritando", "grita", "grito", "preocupada", "preocupado", "susto", "sorprendida", "sorprendido", "angustia"],
        }
        self.ELEMENTS_VOCABULARY = []
        self.ELEMENTS_VARIANT_MAP = {}
        for element, synonyms in self.ELEMENTS_SYNONYMS.items():
            for synonym in synonyms:
                self.ELEMENTS_VOCABULARY.append(synonym)
                self.ELEMENTS_VARIANT_MAP[synonym] = element

    def _normalize(self, text):
        "Normalize the input text by converting it to lowercase and removing accents and diacritics."
        return "".join(
            character
            for character in unicodedata.normalize("NFD", text.lower())
            if unicodedata.category(character) != "Mn"
        )

    def _parse_less_than_100(self, text):
        "Parse numbers less than 100 from the given text."
        return sum(self.UNIDADES.get(part, 0) for part in text.split() if part != "y")

    def _clean_text(self, text):
        "Clean the input text by converting it to lowercase and removing non-alphanumeric characters."
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    def is_real_name(self, name: str, min_rank: int = 1000) -> bool:
        "Check if the given name is a real name based on the names dataset and its rank in Spain."
        info = self.names_dataset.search(name.strip().capitalize())
        if not info:
            return False

        first_name = info.get("first_name")
        if not first_name or not first_name.get("country"):
            return False

        rank = first_name.get("rank", {})
        if not rank:
            return False

        rank_es = rank.get("Spain")
        if rank_es is None:
            return False
        return rank_es <= min_rank

    def extract_names(self, text: str) -> list[str]:
        "Extract real names from the given text."
        tokens = re.findall(r"[a-záéíóúüñ]+", text.lower())
        candidates = [token for token in tokens if len(token) > 2 and token not in self.WORDS_TO_EXCLUDE]
        return [candidate for candidate in candidates if self.is_real_name(candidate)]

    def count_names(self, text: str) -> tuple[int, int]:
        "Count the number of unique and repeated names in the given text."
        names = [name.capitalize() for name in self.extract_names(text)]
        seen = []
        repetitions = 0
        for name in names:
            if name in seen:
                repetitions += 1
            else:
                seen.append(name)
        unique_count = len(seen)
        return unique_count, repetitions

    def score_names(self, text: str) -> int:
        "Score the text based on the number of unique names."   
        unique_count, repetitions = self.count_names(text)
        if unique_count == 0:
            return 0
        if unique_count < 5:
            return 1
        if unique_count < 10:
            return 2
        if unique_count < 15:
            return 3
        if unique_count < 20:
            return 4
        return 5

    def _extract_year(self, text):
        "Extract the year from the given text using various patterns and return the year along with its position in the text if found."
        normalized = self._normalize(text.lower())

        match = re.search(r"(\d{4})", normalized)
        if match:
            return int(match.group(1)), match.span(1)

        match = re.search(r"(dos mil ?[a-záéíóúüñ ]*)", normalized)
        if match:
            value = self.text_to_number(match.group(1))
            if value and 2000 <= value <= 2100:
                return value, match.span(1)

        match = re.search(r"(mil ?[a-záéíóúüñ ]+)", normalized)
        if match:
            value = self.text_to_number(match.group(1))
            if value and 1900 <= value < 2000:
                return value, match.span(1)

        match = re.search(r"veinte\s+(veinti[dn]?os?|d[oi][sz]|tr[ée]s|cuatro|cinco|seis|siete|ocho|nueve)", normalized)
        if match:
            number_text = match.group(1)
            if number_text in self.UNIDADES:
                return 2000 + self.UNIDADES[number_text], match.span(1)

        match = re.search(r"(?:año|del|de)\s*(\d{2})", normalized)
        if match:
            value = int(match.group(1))
            if 0 <= value <= 30:
                return 2000 + value, match.span(1)
            if 90 <= value <= 99:
                return 1900 + value, match.span(1)

        words = re.findall(r"[a-záéíóúñ]+", normalized)
        for index, word in enumerate(words):
            if word in self.UNIDADES and 20 <= self.UNIDADES[word] <= 31:
                context = " ".join(words[max(0, index - 3): index + 3])
                if any(marker in context for marker in ("año", "de", "del", "mes", "es", "el")):
                    return 2000 + self.UNIDADES[word], None
        return None, None

    def _extract_day(self, text, year_span=None):
        "Extract the day from the given text, optionally ignoring a specific span of text that may contain the year."
        normalized = self._normalize(text.lower())
        text_without_year = normalized
        if year_span:
            start, end = year_span
            text_without_year = normalized[:start] + " " * (end - start) + normalized[end:]

        for number in re.findall(r"\d{1,2}", text_without_year):
            if 1 <= int(number) <= 31:
                return int(number)

        words = re.findall(r"[a-záéíóúñ]+(?:\s+y\s+[a-záéíóúñ]+)?", text_without_year)
        for word in words:
            word = word.strip()
            if word in self.UNIDADES and 1 <= self.UNIDADES[word] <= 31:
                return self.UNIDADES[word]

        for key, value in sorted(self.UNIDADES.items(), key=lambda item: -len(item[0])):
            if key in text_without_year and 1 <= value <= 31:
                return value
        return None

    def _extract_month(self, text):
        "Extract the month from the given text and return its index if found."
        normalized = self._normalize(text)
        for month in self.MESES:
            if month in normalized:
                return self.MONTHS_INDEX[month]
        return None

    def _extract_weekday(self, text):
        "Extract the weekday from the given text and return its index if found."
        normalized = self._normalize(text)
        for weekday, value in self.DIAS_SEMANA.items():
            if weekday in normalized:
                return value
        return None

    def text_to_number(self, text):
        "Convert a Spanish number expressed in words to its integer value."
        text = text.replace(" y ", " ")
        tokens = text.strip().lower().split()
        total = 0
        partial = 0
        for token in tokens:
            if token == "mil":
                if partial == 0:
                    partial = 1
                total += partial * 1000
                partial = 0
            elif token in self.NUMBERS_ES:
                partial += self.NUMBERS_ES[token]
        total += partial
        return total

    def _extract_date(self, text):
        "Extract the day, month, and year from the given text."
        year, year_span = self._extract_year(text)
        day = self._extract_day(text, year_span=year_span)
        month = self._extract_month(text)
        return day, month, year

    def analyze_T2(self, text, real_date):
        "Analyze the text for task T2 by extracting the date and comparing it to the real date, scoring based on the accuracy of the extracted information."
        month_gt, day_gt, year_gt = real_date.split("/")
        day_gt = int(day_gt)
        month_gt = int(month_gt)
        year_gt = int(year_gt)
        if year_gt < 100:
            year_gt += 2000

        day, month, year = self._extract_date(text)
        score = 0
        if day == day_gt:
            score += 1
        elif day is None:
            weekday = self._extract_weekday(text)
            if weekday is not None and weekday == date(year_gt, month_gt, day_gt).weekday():
                score += 1
        if month == month_gt:
            score += 1
        if year == year_gt:
            score += 1
        return score

    def analyze_T4_T5(self, text: str) -> int:
        "Analyze the text for tasks T4 and T5 by scoring the names mentioned in the text."
        return self.score_names(text)

    def analyze_T3_T6(self, text: str) -> int:
        "Analyze the text for tasks T3 and T6 by detecting objects mentioned in the text."
        cleaned_text = self._clean_text(text)
        detected_objects = set()
        for word in cleaned_text:
            if len(word) <= 3:
                continue
            match, partial_score, _ = process.extractOne(word, self.VOCABULARY, scorer=fuzz.partial_ratio)
            ratio_score = fuzz.ratio(word, match)
            if partial_score >= 80 and ratio_score >= 80:
                detected_objects.add(self.VARIANT_MAP[match])
        return len(detected_objects)

    def analyze_T7(self, text: str) -> int:
        "Analyze the text for task T7 by detecting elements mentioned in the text."
        cleaned_text = self._clean_text(text)
        detected_elements = set()
        for word in cleaned_text:
            match, score, _ = process.extractOne(word, self.ELEMENTS_VOCABULARY, scorer=fuzz.partial_ratio)
            if score >= 80:
                detected_elements.add(self.ELEMENTS_VARIANT_MAP[match])
        return len(detected_elements)

    def count_interruptions(self, path_label, task):
        "Count the number of interruptions for a specific task in the given label file."
        labels_df = parse_audio_segments(path_label)
        labels_df = labels_df[labels_df[2] == "MED"]
        task_rows = labels_df[labels_df[3] == task]
        return len(task_rows)

    def evaluate_patient(self, row_info, path_label, debug=False) -> dict:
        "Evaluate a single patient's transcription by analyzing the text for each task and counting interruptions, returning a dictionary with the scores and counts."
        text_T2 = row_info["T2"]
        real_date = row_info["Date (mm/dd/yy)"]
        score_T2 = self.analyze_T2(text_T2, real_date)

        text_T3 = row_info["T3"]
        score_T3 = self.analyze_T3_T6(text_T3)

        text_T4 = row_info["T4"]
        score_T4 = self.analyze_T4_T5(text_T4)

        text_T5 = row_info["T5"]
        score_T5 = self.analyze_T4_T5(text_T5)

        text_T6 = row_info["T6"]
        score_T6 = self.analyze_T3_T6(text_T6)
        count_interruptions_T6 = self.count_interruptions(path_label, "T6")

        try:
            text_T7 = row_info["T7"]
            score_T7 = self.analyze_T7(text_T7)
        except Exception:
            text_T7 = ""
            score_T7 = 0

        if debug:
            print("===============================")
            print(f"T2 - Real date: {real_date} - text: {text_T2}")
            print(f"T2 - Score: {score_T2}")
            print("===============================")
            print(f"T3 - text: {text_T3}")
            print(f"T3 - Score: {score_T3}")
            print("===============================")
            print(f"T4 - text: {text_T4}")
            print(f"T4 - Score: {score_T4}")
            print("===============================")
            print(f"T5 - text: {text_T5}")
            print(f"T5 - Score: {score_T5}")
            print("===============================")
            print(f"T6 - text: {text_T6}")
            print(f"T6 - Score: {score_T6}")
            print("===============================")
            print(f"T7 - text: {text_T7}")
            print(f"T7 - Score: {score_T7}")

        return {
            "score_T2": score_T2,
            "score_T3": score_T3,
            "score_T4": score_T4,
            "score_T5": score_T5,
            "score_T6": score_T6,
            "count_interruptions_T6": count_interruptions_T6,
            "score_T7": score_T7,
            "score_total": score_T2 + score_T3 + score_T4 + score_T5 + score_T6 + score_T7,
        }

    def evaluate_transcriptions(self):
        "Evaluate all patient transcriptions by analyzing the text for each task and counting interruptions, returning a DataFrame with the results."
        df_transcriptions = pd.read_csv(PATH_TRANSCRIPTIONS).set_index("id")
        metadata_df = pd.read_csv(PATH_METADATA, sep=";")
        df = df_transcriptions.join(metadata_df.set_index("Participant ID"), how="left")

        relevant_columns = [
            "T1", "T2", "T3", "T4", "T5", "T6", "T7",
            "Group", "Diagnosis (only patients)", "Date (mm/dd/yy)",
        ]
        df = df[relevant_columns]

        results_by_patient = {}
        for patient_id, row in df.iterrows():
            print(f"Evaluating patient {patient_id}...")
            if row["Group"] == "Patient":
                path_label = os.path.join(PATH_DIR_LABELS_PATIENTS, f"{patient_id}.txt")
                label = 1
            else:
                path_label = os.path.join(PATH_DIR_LABELS_CONTROLS, f"{patient_id}.txt")
                label = 0

            results = self.evaluate_patient(row, path_label, debug=False)
            results["id"] = patient_id
            results["label"] = label
            results_by_patient[patient_id] = results

        df_results = pd.DataFrame.from_dict(results_by_patient, orient="index")
        return df_results
