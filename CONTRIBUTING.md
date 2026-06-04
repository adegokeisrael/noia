# Contributing to NOIA

Thank you for your interest in contributing. This project is an
ITU-T FG-AINN Build-a-thon 2025 submission by the AITR-NOIA team.

## Development Setup

```bash
git clone https://github.com/aitr-team/noia-noc-assistant.git
cd noia-noc-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Running Tests

```bash
pytest tests/ -v --cov=noia
```

All PRs must pass the full test suite. Heavy ML components are mocked in the
test suite so no GPU or model downloads are required to run tests.

## Code Style

- Python: follow PEP 8, use type hints on all public functions.
- Docstrings: Google/NumPy style for public classes and methods.
- Maximum line length: 100 characters.

## Branch Naming

- `feature/<short-description>`
- `fix/<short-description>`
- `docs/<short-description>`

## Pull Request Checklist

- [ ] Tests pass: `pytest tests/`
- [ ] Docstrings added/updated
- [ ] `.env.example` updated if new config keys added
- [ ] `CHANGELOG.md` entry added

## License

By contributing, you agree your contributions are licensed under Apache 2.0.
