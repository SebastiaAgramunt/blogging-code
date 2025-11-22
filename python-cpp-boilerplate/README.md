# Python CPP Boilerplate


## Compile and install the package

Create a new package

```bash
rm -rf .venv
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

Then compile C++ bindings and install with pip

```bash
.venv/bin/python -m pip install .
```

## Run an example

To check that this is working try importing the package and running a script

```bash
.venv/bin/python -c "import package_example"
.venv/bin/python scripts/example.py
```

## Tesing

Run tests

```bash
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/pytest -v
```

## Linting

Using ruff

```bash
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/ruff format .

.venv/bin/mypy src
```