"""Classifier: heuristic and learned model agree on clear-cut prompts."""
from conftest import SIMPLE, MEDIUM, COMPLEX
from llm_router.classifier import heuristic_score
def test_heuristic_simple_factoid():
    result = heuristic_score(SIMPLE)
    assert result.label == "simple"
    assert result.method == "heuristic"
def test_heuristic_complex_design_prompt():
    result = heuristic_score(COMPLEX)
    assert result.label == "complex"
def test_learned_classifier_simple(classifier):
    assert classifier.predict(SIMPLE).label == "simple"
def test_learned_classifier_medium(classifier):
    assert classifier.predict(MEDIUM).label == "medium"
def test_learned_classifier_complex(classifier):
    assert classifier.predict(COMPLEX).label == "complex"
def test_classifier_training_accuracy_is_reasonable(classifier):
    assert classifier.train_accuracy() >= 0.8
