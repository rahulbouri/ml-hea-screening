# Contributing to ML-HEA Screening

Thank you for your interest in contributing to this project!

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs
- Include Python version, OS, and relevant error messages
- Provide minimal reproducible examples

### Submitting Changes
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes with clear commit messages
4. Ensure code passes all tests: `python test_setup.py`
5. Submit a pull request with a clear description

### Code Style
- Follow PEP 8 for Python code
- Use descriptive variable and function names
- Add docstrings to public functions
- Include comments for complex logic

### Testing
Before submitting, ensure:
1. All dependencies are installed: `pip install -r requirements.txt`
2. Setup verification passes: `python test_setup.py`
3. No breaking changes to existing APIs

### Documentation
- Update README.md if adding new features
- Document data format changes
- Include usage examples

## Development Setup

```bash
# Clone your fork
git clone <your-fork-url>
cd ml-hea-screening

# Create development environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install development tools (optional)
pip install black flake8 mypy
```

## Areas for Contribution

- Additional feature engineering methods
- New model architectures
- Performance optimizations
- Documentation improvements
- Dataset expansion
- Visualization tools
- Hyperparameter optimization

## Questions?

Create an issue or discussion thread on GitHub.

Thank you for contributing!
