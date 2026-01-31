# Contributing to ML Project Framework

Thank you for your interest in contributing to the ML Project Framework! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Support fellow contributors

## Getting Started

1. **Fork the repository** - Create your own copy
2. **Clone your fork** - `git clone https://github.com/YOUR_USERNAME/ml.git`
3. **Create a branch** - `git checkout -b feature/your-feature-name`
4. **Set up development environment** - `pip install -r requirements.txt && pip install -e .`

## Development Workflow

### Before Starting
- Check existing issues and pull requests
- Discuss major changes by opening an issue first

### Making Changes
1. Create a feature branch from `main`
2. Write clear, descriptive commit messages
3. Keep commits focused and logical
4. Add tests for new functionality
5. Update documentation as needed

### Code Style
- Follow PEP 8 conventions
- Use type hints where applicable
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Testing
```bash
# Run tests
python -m pytest tests/

# Check code style
python -m pylint src/

# Check type hints
python -m mypy src/
```

### Documentation
- Update README.md for user-facing changes
- Add inline comments for complex logic
- Update QUICKSTART.md for new features
- Include examples in docstrings

## Submitting Changes

### Commit Messages
```
[TYPE] Brief description of change

Longer explanation of what and why, if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `perf`, `chore`

### Pull Requests
1. Provide a clear title and description
2. Reference related issues (#123)
3. Include any breaking changes
4. Request review from maintainers

## Areas for Contribution

### Code
- New algorithms (XGBoost, LightGBM, Neural Networks)
- Performance optimizations
- Bug fixes and issue resolutions
- New evaluation metrics

### Documentation
- Tutorial notebooks
- Example datasets
- API documentation
- Troubleshooting guides

### Testing
- Unit tests for modules
- Integration tests
- Test coverage improvements

## Reporting Issues

**Bug Reports** should include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant code snippet

**Feature Requests** should include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (if any)
- Examples of similar features

## Review Process

1. Maintainers review your PR
2. Address feedback and make updates
3. Once approved, your PR will be merged
4. Your contribution will be included in the next release

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## Questions?

- Check the [QUICKSTART.md](QUICKSTART.md) guide
- Review existing [issues](../../issues)
- Open a new discussion for questions

---

**Thank you for contributing to ML Project Framework!** Your efforts help make this project better for everyone. 🙏
