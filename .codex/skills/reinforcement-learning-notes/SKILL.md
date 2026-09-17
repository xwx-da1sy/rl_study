---
name: reinforcement-learning-notes
description: Write focused, beginner-friendly reinforcement learning lessons as Jupyter notebooks in this repository.
---

# Reinforcement Learning Notes

Use this skill when starting or continuing a reinforcement learning lesson in this repository.

## Learning style

- Follow the progression in `background_information/00_学习导航.ipynb`: GridWorld and returns, then tabular methods, DQN, and policy methods.
- Introduce one core question per notebook. Start with what the concept means and why it is needed, then work through a small example, summarize, and end with self-check questions.
- Connect to the learner's NumPy and PyTorch experience only where it clarifies the new concept.
- Keep notebooks reading-focused. Use only small code cells when a quick observation supports understanding.
- Do not create scripts, model files, utilities, or code packages unless the user explicitly asks for implementation.
- Write original Chinese explanations. Avoid jumping ahead to training loops or APIs before the current concept is clear.

## Repository layout

- Keep the navigation in `background_information/` and basic lessons in `rl_basics/`.
- Do not create additional folders or Python code packages unless the user asks for them.
- Do not create standalone Markdown lesson files.

## Formulas and validation

- In notebook Markdown cells, use Jupyter-renderable LaTeX for mathematical notation and put display `$$` delimiters on separate lines.
- Check worktree status before editing and preserve unrelated changes.
- Validate each notebook as JSON and execute its code cells in order. When available, export it with `jupyter nbconvert` to check rendering.
- Use the local `render-notebook-formulas` skill and its audit script when changing formulas.
