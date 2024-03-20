default:
	@just --choose

rs_build:
	cargo build --release

rs_docs:
	cargo doc

rs_docs_open:
	cargo doc --open

py_build:
	#!/usr/bin/env bash
	source .venv/bin/activate
	maturin dev --release

py_docs_test: py_build
	#!/usr/bin/env bash
	source .venv/bin/activate
	python -m doctest -v docs/source/index.rst

py_docs_build: py_build py_docs_test
	#!/usr/bin/env bash
	source .venv/bin/activate
	cd docs
	make clean
	make html

py_docs_open: py_docs_build
	xdg-open docs/build/html/index.html

setup_venv:
	#!/usr/bin/env bash
	rm -rf .venv
	python -m venv .venv
	source .venv/bin/activate
	pip install maturin[zig]
	pip install -r docs/requirements.txt
	# Install requirements for celegans example
	pip install numpy pandas networkx
