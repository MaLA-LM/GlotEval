import re

LANG_TO_ANSWER_PREFIX = {
    "en": "Answer",
    "ar": "الإجابة",
    "bn": "উত্তর",
    "cs": "Odpověď",
    "de": "Antwort",
    "es": "Respuesta",
    "fr": "Réponse",
    "hu": "Válasz",
    "ja": "答え",
    "ko": "정답",
    "ru": "Ответ",
    "sr": "Одговор",
    "sw": "Jibu",
    "te": "సమాధానం",
    "th": "คำตอบ",
    "vi": "Đáp án",
    "zh": "答案",
}


def parse_answer(answer: str, answer_prefix: str) -> str:
    if answer_prefix not in answer:
        match = list(re.finditer(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", answer))
        if match:
            answer_text = match[-1].group()
        else:
            return ""
    else:
        answer_text = answer.split(answer_prefix)[-1].strip()

    # find all the numbers (including decimals) in the string
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", "").replace("$", ""))

    # return the first number (removing trailing decimal point if present),
    # or an empty string if there were no numbers
    return numbers[-1].rstrip(".") if numbers else ""


def score_mgsm(target: str, prediction: str) -> bool:
    if "." in prediction:
        prediction = prediction.rstrip("0").rstrip(".")

    target = target.replace(",", "")
    prediction = prediction.replace(",", "")

    return target == prediction


def compute_math_acc(dataset: list, responses: list, lang: str):
    answer_prefix = LANG_TO_ANSWER_PREFIX[lang]
    extracted_answers = [parse_answer(r, answer_prefix) for r in responses]
    correct = sum(score_mgsm(str(example["answer_number"]), ans) for ans, example in zip(extracted_answers, dataset))
    return {"accuracy": correct / len(dataset) if dataset else 0}