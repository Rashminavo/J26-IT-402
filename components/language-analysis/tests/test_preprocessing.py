import pytest

from language_analysis.preprocessing import (
    clean_text,
    preprocess_text,
    tokenize_text,
)


def test_clean_text_removes_url_and_normalizes_whitespace():
    text = "Hello   world!\nVisit https://example.com now."

    result = clean_text(text)

    assert "https://example.com" not in result
    assert result == "Hello world! Visit now."


def test_tokenize_text_returns_lowercase_tokens():
    result = tokenize_text("Hello World!")

    assert result[:2] == ["hello", "world"]


def test_preprocess_empty_text():
    result = preprocess_text("")

    assert result["cleaned_text"] == ""
    assert result["tokens"] == []
    assert result["token_count"] == 0


def test_preprocess_rejects_non_string():
    with pytest.raises(TypeError):
        preprocess_text(123)


def test_clean_text_decodes_html_entities():
    result = clean_text("I&#39;m not okay &amp; tired")

    assert result == "I'm not okay & tired"


def test_tokenize_preserves_negation_words():
    result = tokenize_text("I am not happy and never feel okay")

    assert "not" in result
    assert "never" in result


def test_preprocess_whitespace_only_text():
    result = preprocess_text("   \n\t  ")

    assert result["cleaned_text"] == ""
    assert result["tokens"] == []
    assert result["token_count"] == 0
