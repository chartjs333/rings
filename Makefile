.PHONY: test demo
test:
	PYTHONPATH=src python -m unittest discover -s tests -v
demo:
	PYTHONPATH=src python -m rings_fly demo --out .runtime/toy_demo
