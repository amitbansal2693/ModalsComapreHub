# 🤖 OpenAI GPT Model Comparison Tool

A **Streamlit web app** to compare responses, token usage, and cost across multiple OpenAI GPT models.  
Supports both older (`gpt-3.5`, `gpt-4`) and newer (`gpt-5-nano`, `gpt-4o`) models with correct handling of API parameter differences.

**Author**: Amit Kumar Bansal  
---

## ✨ Features

- Compare **multiple models side by side**:
  - `gpt-4`
  - `gpt-3.5-turbo`
  - `gpt-5-nano-2025-08-07`
  - `gpt-4o-mini`
  - `gpt-3.5-turbo-16k`
- Track **tokens** (prompt, completion, total).
- Estimate **cost per response** based on OpenAI pricing.
- Gracefully **skips errors** (one model failing won’t block others).
- Handles **strict models** (like `gpt-5-nano`) by simplifying API calls automatically.
- Provides **moderation checks** for input prompts.

---

## 🛠️ Requirements

- Python 3.9+
- OpenAI API key ([Get one here](https://platform.openai.com/account/api-keys))

Install dependencies:

```bash
pip install -r requirements.txt
