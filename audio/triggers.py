# audio/triggers.py
from __future__ import annotations
import csv
from typing import List, Dict

# Небольшая карта переводов для самых частых раздражителей.
# Остальные будут показаны как в оригинале (ENG), чтобы ничего не падало.
_RU_TOKEN_MAP = {
    "chew": "чавканье",
    "chewing": "чавканье",
    "slurp": "сёрбанье",
    "gulp": "глоток",
    "swallow": "глотание",
    "lick": "лизание",
    "kiss": "поцелуй",
    "mouth": "рот (звуки)",
    "breathe": "дыхание",
    "breathing": "дыхание",
    "cough": "кашель",
    "sneeze": "чихание",
    "hiccup": "икота",
    "sniff": "шмыганье",
    "snore": "храп",
    "typing": "печать на клавиатуре",
    "keyboard": "клавиатура",
    "click": "щелчок",
    "tick": "тик-так",
    "tap": "стук",
    "knock": "стук",
    "clap": "хлопок",
    "clatter": "лязг",
    "cutlery": "столовые приборы",
    "glass": "стекло (звук)",
    "plate": "тарелка",
    "spoon": "ложка",
    "fork": "вилка",
    "knife": "нож",
    "pen": "ручка (щелчки)",
    "pencil": "карандаш",
    "nail": "ногти (щёлканье/стук)",
    "finger": "пальцы (щелчки)",
    "tap water": "капли/вода",
    "drip": "капли",
    "ring": "звон",
    "bell": "колокол/звонок",
    "alarm": "сигнал/будильник",
}

# fallback на случай полной недоступности модели/CSV
_FALLBACK = [
    "chewing", "slurp", "gulp", "lick", "mouth sounds", "breathing",
    "cough", "sneeze", "sniff", "snore",
    "typing", "keyboard", "mouse click", "tick-tock", "tap", "knock",
    "clap", "cutlery", "glass", "plate", "spoon", "fork", "knife",
]

def _to_ru(name: str) -> str:
    low = name.lower()
    for k, v in _RU_TOKEN_MAP.items():
        if k in low:
            return v
    # чуть улучшим читаемость EN
    return name.replace("_", " ")

def load_yamnet_labels() -> List[str]:
    """Пробуем достать class_map у локально закешированного YAMNet."""
    try:
        import tensorflow_hub as hub
        m = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = m.class_map_path().numpy().decode("utf-8")
        labels: List[str] = []
        with open(class_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(row.get("display_name") or row.get("name") or "")
        # фильтруем пустые
        return [x for x in labels if x]
    except Exception:
        # если не вышло — отдаём fallback
        return list(_FALLBACK)

def labels_ru() -> List[Dict]:
    """Собираем [{idx, en, ru}] для фронта."""
    en = load_yamnet_labels()
    out = []
    for i, name in enumerate(en):
        out.append({
            "idx": i,
            "en": name,
            "ru": _to_ru(name),
        })
    return out
