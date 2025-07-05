import re

def parse_answer(answer: str) -> str:
    match = re.findall(r"\(([A-Z])\)", answer)
    if match:
        answer_text = match[-1]
    else:
        return ""

    return answer_text


def compute_science_acc(dataset: list, responses: list, lang: str):
    extracted_answers = [parse_answer(r) for r in responses]
    correct = sum(ans == example["answer"] for ans, example in zip(extracted_answers, dataset))
    return {"accuracy": correct / len(dataset) if dataset else 0}