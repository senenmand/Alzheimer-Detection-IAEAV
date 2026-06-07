"""
config.py — Global constants, prompts, and CUDA configuration.

Set the HF_TOKEN environment variable with your Hugging Face token
before running the pipeline:
    export HF_TOKEN="hf_..."
"""
import os

import torch
from dotenv import load_dotenv


load_dotenv()

RATE = 16_000
BATCH_SIZE = 8
W2V_DIM = 100
BERT_DIM = 768
HUBERT_DIM = 768

AUTH_TOKEN_HF = os.environ.get("HF_TOKEN", "")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_BASE = (
    "Entrevista clínica neuropsicológica en español. "
    "El paciente habla en español de España o Latinoamérica. "
)

PROMPTS_TAREA = {
    "T1": PROMPT_BASE + (
        "El paciente describe cómo es su memoria y qué cosas le cuesta más recordar. "
        "Vocabulario esperado: memoria, recuerdo, olvidar, nombres, fechas, pastillas, "
        "lista de la compra, despiste, concentración."
    ),
    "T2": PROMPT_BASE + (
        "El paciente dice la fecha de hoy: día de la semana, día del mes, mes y año. "
        "Vocabulario esperado: lunes, martes, miércoles, jueves, viernes, sábado, domingo, "
        "enero, febrero, marzo, abril, mayo, junio, julio, agosto, septiembre, octubre, "
        "noviembre, diciembre, dos mil veinticinco."
    ),
    "T3": PROMPT_BASE + (
        "El paciente nombra en voz alta seis objetos cotidianos que aparecen en pantalla. "
        "Vocabulario esperado: baraja, coche, pera, trompeta, zapatos y cuchara"
    ),
    "T4": PROMPT_BASE + (
        "El paciente enumera nombres propios de hombre o mujer en 30 segundos, separados por pausas. "
        "Vocabulario esperado: nombres masculinos o femeninos en español"
    ),
    "T5": PROMPT_BASE + (
        "El paciente enumera nombres propios de hombre o mujer en 30 segundos, separados por pausas. "
        "Vocabulario esperado: nombres masculinos o femeninos en español"
    ),
    "T6": PROMPT_BASE + (
        "El paciente intenta recordar y nombrar los seis objetos que vio anteriormente. "
        "Vocabulario esperado: baraja, coche, pera, trompeta, zapatos y cuchara"
    ),
    "T7": PROMPT_BASE + (
        "El paciente describe en detalle una imagen: personas, acciones, objetos y escenario. En concreto describe una escena de una calle donde un niño juega con una pelota roja cerca de un coche, y el riesgo de que el coche atropelle al niño. "
        "Vocabulario esperado: hay, veo, parece, está, están, una persona, "
        "una mujer, niño, niña, chaqueta amarilla, sentado, de pie, cogiendo, mirando, fondo, "
        "calle, pelota, roja, coche, atropellar, accidente"
    ),
}

PROMPT_GENERIC = PROMPT_BASE + (
    "El paciente responde preguntas sobre memoria, orientación y cognición."
)
