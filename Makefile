.PHONY: test clean

clean:
	-rm -r .req .pytest_cache dpsbin/__pycache__

.req: requirements.txt
	python -m pip install -r requirements.txt
	touch .req

test: .req
	pytest
