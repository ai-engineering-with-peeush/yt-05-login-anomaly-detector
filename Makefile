.PHONY: install data eval serve

PYTHON = .venv\Scripts\python

install:
	@if not exist .venv uv venv
	uv pip install -r requirements.txt

data:
	$(PYTHON) src/generate_logs.py

eval:
	$(PYTHON) evals/run_eval.py

serve:
	$(PYTHON) -m uvicorn serve:app --reload
