"""Original SQuAD-style token F1 for MASBench Reasoning."""
import re, string
from collections import Counter

def _squad_f1(reference: str, prediction: str) -> float:
    def tokens(text: str) -> list[str]:
        text = text.lower()
        text = "".join(char for char in text if char not in string.punctuation)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return " ".join(text.split()).split()

    ref, pred = tokens(reference), tokens(prediction)
    if not ref or not pred:
        return float(ref == pred)
    common = Counter(ref) & Counter(pred)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)
