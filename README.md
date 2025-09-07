# clock_rl
Training model for clock recognition using reinforcement learning with Qwen2.5-VL.

## Setup with uv

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Installation

1. Install uv:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create virtual environment and install dependencies:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

### Usage

- Run training: `python trainer.py`
- Run tests: `python test.py`
- Run evaluation: `python eval.py`

### Development

- Add a new dependency: `uv add package-name`
- Add a development dependency: `uv add --dev package-name`
- Update dependencies: `uv sync --upgrade`
- Run with different Python version: `uv run --python 3.11 python trainer.py`

### Environment Setup

Make sure to create a `.env` file with your Hugging Face token:
```
HF_TOKEN=your_hugging_face_token_here
```
