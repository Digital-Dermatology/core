# Core Library
Library for helpful stuff.

Run `make` for a list of possible targets.

## Installation
Run this command for installation
`make install`

## Code and test conventions
- `black` for code style
- `isort` for import sorting
- docstring style: `sphinx`
- `pytest` for running tests

### Development installation and configurations
To set up your dev environment run:
```bash
pip install -r requirements.txt
# Install pre-commit hook
pre-commit install
```
To run all the linters on all files:
```bash
pre-commit run --all-files
```
