import html
import re

from nltk.tokenize import wordpunct_tokenize


_URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Perform conservative text cleaning before downstream NLP analysis."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = html.unescape(text)
    text = _URL_PATTERN.sub(" ", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = _WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


def tokenize_text(text: str) -> list[str]:
    """Tokenize cleaned text with NLTK wordpunct_tokenize."""
    cleaned = clean_text(text)

    if not cleaned:
        return []

    raw_tokens = wordpunct_tokenize(cleaned.lower())

    tokens: list[str] = []

    for token in raw_tokens:
        normalized = token.strip()

        if not normalized:
            continue

        if any(character.isalnum() for character in normalized):
            tokens.append(normalized)

    return tokens


def preprocess_text(text: str) -> dict[str, object]:
    """Return cleaned text and basic token metadata."""
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)

    return {
        "cleaned_text": cleaned,
        "tokens": tokens,
        "token_count": len(tokens),
    }