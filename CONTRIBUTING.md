# Contributing to ResumeBench

Thanks for your interest in contributing. Here's how you can help.

## Report Issues

Found a scoring bug, schema issue, or incorrect ground truth? [Open an issue](https://github.com/careerflow/ResumeBench/issues) with:

- What you expected vs what happened
- The pipeline and split you ran
- Any relevant output or error messages

## Code Changes

For bug fixes or improvements to the scoring engine, CLI, or providers:

1. Fork the repo and create a branch from `main`
2. Install dependencies: `pip install -e ".[dev,runners]"`
3. Make your changes
4. Run tests: `pytest --tb=short -q`
5. Run lint: `ruff check src/ tests/`
6. Open a PR against `main`

Keep PRs focused on a single change. If you're planning something larger, open an issue first so we can discuss the approach.

## Add a Provider

See the "Adding a custom provider" section in the [README](README.md#usage). If you've built a provider for a publicly accessible extraction system, we'd welcome a PR adding it.

## Style

- We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Run `ruff format src/ tests/` before committing
- Tests go in `tests/unit/`

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
