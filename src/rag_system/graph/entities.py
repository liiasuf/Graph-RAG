from __future__ import annotations

import re

_STOPWORDS = {
    "A", "An", "The",
    "And", "Or", "But", "Nor", "For", "So", "Yet",
    "In", "On", "At", "Of", "To", "By", "As", "With", "From", "Into", "Onto",
    "Over", "Under", "After", "Before", "During", "Since", "Until", "While",
    "It", "Its", "This", "That", "These", "Those",
    "He", "She", "They", "We", "I", "You", "His", "Her", "Their", "Our", "Your",
    "Is", "Was", "Are", "Were", "Be", "Been", "Being",
    "Has", "Have", "Had", "Will", "Would", "Can", "Could", "May", "Might", "Shall", "Should",
    "Not", "If", "Than", "Then", "Also", "However", "Although", "Because",
}

# Matches runs of one or more capitalised "words" separated by single spaces.
# A "word" starts with an uppercase letter and may contain letters/digits and
# an internal apostrophe/period (e.g. "O'Brien", "U.S.").
_PROPER_NOUN_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:[.'][A-Za-z]+)?(?:\s+[A-Z][A-Za-z0-9]*(?:[.'][A-Za-z]+)?)*\b"
)


def normalize_entity(text: str) -> str:
    """
    Normalize an entity mention to a canonical lowercase key used to
    identify graph nodes (whitespace-collapsed, lowercased).
    """

    return " ".join(text.lower().split())


def extract_entities(text: str, *, max_words: int = 5, min_chars: int = 3) -> set[str]:
    """
    Extract candidate named-entity mentions from `text` using a lightweight,
    dependency-free heuristic: runs of consecutive capitalised words.

    Leading/trailing stopwords (sentence-initial "The", pronouns, etc.) are
    trimmed from each match. This is intentionally simple - it favours speed
    and zero extra dependencies over the recall/precision of a full NER
    model, which is sufficient for building a co-occurrence graph over
    Wikipedia-style text such as HotpotQA.

    Returns the original-cased mention strings (whitespace-normalised).
    """

    entities: set[str] = set()

    for m in _PROPER_NOUN_RE.finditer(text):
        words = m.group(0).split()

        while words and words[0] in _STOPWORDS:
            words = words[1:]
        while words and words[-1] in _STOPWORDS:
            words = words[:-1]

        if not words or len(words) > max_words:
            continue

        mention = " ".join(words)
        if len(mention) < min_chars:
            continue

        entities.add(mention)

    return entities
