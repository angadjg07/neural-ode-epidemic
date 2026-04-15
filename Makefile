.PHONY: setup train evaluate dashboard results test clean

setup:
	@echo "Setting up environment..."
	python -m venv venv
	./venv/bin/pip install -r requirements.txt

train:
	@echo "Training models..."
	./venv/bin/python src/train.py

evaluate:
	@echo "Evaluating models..."
	./venv/bin/python src/evaluate.py

dashboard:
	@echo "Starting dashboard..."
	./venv/bin/streamlit run app/main.py

results:
	@echo "Building results page..."
	./venv/bin/python results/build.py

pipeline:
	@echo "Running end-to-end evaluation pipeline..."
	./venv/bin/python src/evaluate.py
	./venv/bin/python results/build.py
	open results/index.html

test:
	@echo "Running tests..."
	./venv/bin/pytest tests/

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache src/__pycache__
