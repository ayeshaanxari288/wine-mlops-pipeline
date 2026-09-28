.PHONY: install lint test train clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest -v

train:
	python src/train.py

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache mlruns *.pyc
```[cite: 2]