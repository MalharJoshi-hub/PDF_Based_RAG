"""
Unit tests for PromptBuilder module.
"""

from src.generation.prompt_templates import PromptBuilder


def test_extract_key_points():
    text = (
        "Hibernate is an Object-Relational Mapping (ORM) solution for Java. "
        "It provides transparent persistence for plain old Java objects. "
        "Short phrase. "
        "The application configures database settings via hibernate.cfg.xml."
    )
    points = PromptBuilder.extract_key_points("What is Hibernate?", text, max_points=2)
    assert len(points) >= 1
    assert any("Hibernate is an Object-Relational Mapping" in p for p in points)


def test_extract_key_points_empty():
    assert PromptBuilder.extract_key_points("query", "") == []


def test_build_qa_prompt():
    chunks = [
        {"text": "Hibernate provides data persistence mapping objects to relational database tables."}
    ]
    prompt = PromptBuilder.build_qa_prompt("What is Hibernate?", chunks, template_style="instruction")
    assert "Context information:" in prompt
    assert "What is Hibernate?" in prompt
    assert "Hibernate provides data persistence" in prompt


def test_build_qa_prompt_styles():
    chunks = [{"text": "Sample context text."}]
    p_sum = PromptBuilder.build_qa_prompt("Question", chunks, template_style="summarization")
    assert "Summarize the answer clearly" in p_sum

    p_dir = PromptBuilder.build_qa_prompt("Question", chunks, template_style="direct")
    assert "What is the explanation for: Question" in p_dir
