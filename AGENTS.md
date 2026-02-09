# codex.md — LLM from scratch project instructions

## Project goal
Build a decoder-only Transformer language model *from scratch*.
"From scratch" means:
- No PyTorch / TensorFlow / JAX
- No HuggingFace Transformers
- NumPy is allowed for array storage and basic ops
- Everything else (autograd, layers, attention, training loop, tokenizer) must be implemented in this repo.

Primary objective:
- Clarity and correctness over performance.

## Repository structure (must follow)
- src/
  - tensor/        # Tensor + autograd core
  - nn/            # Linear, LayerNorm, Embedding, Dropout
  - attention/     # scaled dot-product attention, multi-head, causal mask
  - transformer/   # Transformer blocks + GPT model
  - tokenizer/     # BPE tokenizer
  - training/      # optimizer, lr schedule, checkpointing
  - sampling/      # generation (temperature, top-k/top-p)
- tests/           # pytest unit tests
- demos/           # runnable demos
- docs/            # notes and design docs

## Non-negotiable constraints
1. Every new component must include:
   - unit tests in `tests/`
   - a small runnable demo in `demos/` when appropriate
2. After making changes, always run:
   - `python -m pytest -q`
3. Keep APIs small and explicit.
4. Do not introduce heavy dependencies.

## Coding style
- Python 3.11+
- Type hints where helpful
- Prefer small, readable functions
- Add docstrings to core primitives (Tensor, backward, attention)

## Definition of Done (for any task)
A task is done only if:
- tests pass (`pytest`)
- demo script runs end-to-end (if required)
- code is formatted and readable
- you explain *what changed* and *how to verify*

## Current milestone (START HERE)
Milestone 1: Minimal autograd engine

### Deliverables
- `src/tensor/tensor.py`:
  - Tensor(data, requires_grad)
  - ops: add, mul, matmul, sum, mean, relu
  - backward() using reverse topological order
- `tests/test_gradcheck.py`:
  - finite-difference gradient check for each op
- Optional: `demos/demo_mlp.py`:
  - tiny MLP trained on a toy dataset; loss must decrease

### Notes
- Gradients must accumulate (+=) correctly.
- Use deterministic random seeds in tests/demos.
