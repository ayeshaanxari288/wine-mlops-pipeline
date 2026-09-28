import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model


def evaluate():
    model, accuracy = train_model()
    if accuracy < 0.85:
        raise ValueError("Model performance below threshold of 0.85")
    print(f"Model evaluation passed with accuracy: {accuracy}")


if __name__ == "__main__":
    evaluate()