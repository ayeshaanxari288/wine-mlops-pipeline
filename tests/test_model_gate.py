from src.train import train_model


def test_model_accuracy_gate():
    model, accuracy = train_model()
    assert accuracy >= 0.85