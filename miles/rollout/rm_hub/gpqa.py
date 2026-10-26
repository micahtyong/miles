import re
import string
from collections.abc import Iterable

DEFAULT_VALID_LETTERS = list(string.ascii_uppercase[:8])


def _strip_chain_of_thought(text: str) -> str:
    if not text:
        return ""

    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]

    return text


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# Keywords match case-insensitively, but the option itself must be a standalone capital
# letter, so words such as "definitely" or the article "A" are never read as options.
_LETTER = r"\(?([A-Z])\)?(?![A-Za-z0-9])"
_LETTER_PATTERNS = [
    r"(?i:final\s*(?:answer|option))\s*(?i:is|:)?\s*(?:[a-z]+\s+)?" + _LETTER,
    r"(?i:answer|option|choice)\s*(?i:is|:)?\s*(?:[a-z]+\s+)?" + _LETTER,
    r"\b([A-Z])\s*(?i:is\s*(?:the\s*)?correct)",
]


def _extract_letter_from_response(response: str, valid_letters: Iterable[str]) -> str | None:
    """
    Extract the single option letter the model committed to, or None if it did not commit.
    """
    if not response:
        return None

    text = _strip_chain_of_thought(response)
    valid_letters = {letter.upper() for letter in valid_letters}
    for pattern in _LETTER_PATTERNS:
        match = re.search(pattern, text)
        if match and match.group(1) in valid_letters:
            return match.group(1)

    # Fallback: the whole answer is a bare option letter, e.g. "B", "(B)" or "B.".
    match = re.fullmatch(r"\W*\(?([A-Z])\)?\W*", text.strip())
    if match and match.group(1) in valid_letters:
        return match.group(1)

    return None


def _unique_choice_letter(response: str, choices: list, valid_letters: list[str]) -> str | None:
    """Return the letter of the only choice whose text appears in the response, if exactly one does."""
    normalized_response = f" {_normalize_text(_strip_chain_of_thought(response))} "
    matched = [
        letter
        for letter, choice in zip(valid_letters, choices, strict=False)
        if _normalize_text(str(choice)) and f" {_normalize_text(str(choice))} " in normalized_response
    ]
    return matched[0] if len(matched) == 1 else None


def compute_gpqa_reward(response: str, label, metadata: dict | None = None) -> float:
    """Rule-based scorer for GPQA-style multiple-choice evaluation."""
    if response is None:
        return 0.0

    metadata = metadata or {}

    choices = metadata.get("choices")
    if isinstance(choices, dict):
        choices = list(choices.values())
    elif choices is not None:
        choices = list(choices)

    valid_letters = metadata.get("valid_letters")
    if valid_letters:
        valid_letters = [str(letter).upper() for letter in valid_letters]
    elif choices:
        valid_letters = list(string.ascii_uppercase[: len(choices)])
    else:
        valid_letters = DEFAULT_VALID_LETTERS

    correct_letter = metadata.get("correct_letter")
    if isinstance(correct_letter, str):
        correct_letter = correct_letter.strip().upper()
    else:
        correct_letter = None

    label_text = None
    if isinstance(label, str):
        label_text = label.strip()
        if len(label_text) == 1 and label_text.upper() in valid_letters and not correct_letter:
            correct_letter = label_text.upper()
    elif isinstance(label, (int, float)):
        idx = int(label)
        if 0 <= idx < len(valid_letters):
            correct_letter = valid_letters[idx]

    if not correct_letter and choices and label_text:
        normalized_label = _normalize_text(label_text)
        for idx, choice in enumerate(choices):
            if _normalize_text(str(choice)) == normalized_label:
                correct_letter = valid_letters[idx]
                metadata.setdefault("correct_answer", choice)
                break

    extracted_letter = _extract_letter_from_response(response, valid_letters)
    if extracted_letter and correct_letter:
        return 1.0 if extracted_letter == correct_letter else 0.0

    # Without a committed letter, credit the answer text only when it names exactly one option.
    # A response that lists several options is a hedge and scores 0.
    if correct_letter and choices:
        return 1.0 if _unique_choice_letter(response, choices, valid_letters) == correct_letter else 0.0

    # Single-letter labels are option letters, never answer text: a substring test on them
    # would credit any response containing that character.
    candidate_answers = [str(metadata[key]) for key in ("correct_answer", "answer_text") if metadata.get(key)]
    if label_text and not correct_letter:
        candidate_answers.append(label_text)

    normalized_targets = {_normalize_text(text) for text in candidate_answers if len(text.strip()) > 1}
    normalized_response = f" {_normalize_text(_strip_chain_of_thought(response))} "
    for target in normalized_targets:
        if target and f" {target} " in normalized_response:
            return 1.0

    if extracted_letter and not correct_letter and label_text:
        return 1.0 if extracted_letter == label_text.strip().upper() else 0.0

    return 0.0
