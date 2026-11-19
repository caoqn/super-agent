"""GAIA official quasi-exact-match scorer (same logic as the leaderboard's question_scorer)."""
import re
import string


def normalize_number_str(number_str):
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def split_string(s, char_list=(",", ";")):
    return re.split(f"[{''.join(char_list)}]", s)


def normalize_str(input_str, remove_punct=True):
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        return no_spaces.lower().translate(str.maketrans("", "", string.punctuation))
    return no_spaces.lower()


def _is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def question_scorer(model_answer, ground_truth):
    if model_answer is None:
        model_answer = "None"
    model_answer, ground_truth = str(model_answer), str(ground_truth)
    if _is_float(ground_truth):
        return normalize_number_str(model_answer) == float(ground_truth)
    if any(c in ground_truth for c in [",", ";"]):
        gt, ma = split_string(ground_truth), split_string(model_answer)
        if len(gt) != len(ma):
            return False
        ok = []
        for m, g in zip(ma, gt):
            if _is_float(g):
                ok.append(normalize_number_str(m) == float(g))
            else:
                ok.append(normalize_str(m, remove_punct=False) == normalize_str(g, remove_punct=False))
        return all(ok)
    return normalize_str(model_answer) == normalize_str(ground_truth)
