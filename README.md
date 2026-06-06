# PersonaSteer

PersonaSteer is a personalization model for transformer-based text generation that conditions generation on a user's profile and task instruction.

## Method Overview

<p align="center">
  <img src="assets/main_00.png" alt="PersonaSteer method diagram" width="90%">
</p>

*Method diagram: dual-stream user encoding, adaptive fusion, and dynamic steering at a target transformer layer. [PDF version](assets/main.pdf)*

| Document | Contents |
|----------|----------|
| [README_en.md](README_en.md) | Full setup, data formats, training, and evaluation |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture overview (Chinese) |

## Quick start

```bash
pip install -r requirements.txt
pip install -r requirements-eval.txt
cp .env.example .env   # optional: API keys for data prep / LLM judge
python scripts/smoke_check.py
```

See [README_en.md](README_en.md) for the complete open-source workflow.
