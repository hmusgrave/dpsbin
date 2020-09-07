.PHONY: test clean

clean:
	-rm -rf .eggs .mutmut-cache .req .pytest_cache/ dpsbin/__pycache__/ dist/ build/ dpsbin.egg-info/

.req: requirements.txt
	python -m pip install -r requirements.txt
	touch .req

test: .req
	pytest
