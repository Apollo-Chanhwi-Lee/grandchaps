"""Shared prompt format for the LM fine-tune (training data prep and inference must both use this).

Contract: a training record is {"prompt": input_text, "completion": target_text} (mlx-lm
CompletionsDataset).  mlx-lm wraps it as messages [user=prompt, assistant=completion] and renders the
Qwen3 chat template.  Inference must feed the model exactly the same text up to the completion start:

    build_prompt(input_text, tokenizer)
      == "<|im_start|>user\\n{input}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"

i.e. thinking is disabled (empty think block) consistently.  The training text is that prompt followed
by the target and "<|im_end|>\\n".  Generation should stop at "<|im_end|>" (see clean_completion).

Masking caveat: `--mask-prompt` starts the loss 4 tokens BEFORE the end of this inference prompt, so the empty think
block "<think>\\n\\n</think>\\n\\n" is inside the loss (the user input is not).  The inference prompt is a token prefix
of the training text, but it is not "exactly masked" (tests/test_prep_mlx.py::test_real_matches_mlx_completions_dataset).
"""
from __future__ import annotations

MODEL_ID = "Qwen/Qwen3-1.7B"
END_TOKENS = ("<|im_end|>", "<|endoftext|>")


def build_messages(input_text: str) -> list[dict]:
    """Chat messages for one request (no assistant turn)."""
    return [{"role": "user", "content": input_text}]


def build_prompt(input_text: str, tokenizer) -> str:
    """Prompt string ending exactly where the completion starts (thinking disabled)."""
    return tokenizer.apply_chat_template(
        build_messages(input_text), tokenize=False, add_generation_prompt=True,
        enable_thinking=False)


def build_training_record(input_text: str, target_text: str) -> dict:
    """mlx-lm {"prompt","completion"} record; mlx-lm renders the chat template itself."""
    return {"prompt": input_text, "completion": target_text}


def build_full_text(input_text: str, target_text: str, tokenizer) -> str:
    """Text mlx-lm trains on for one record (chat template over user + assistant turns)."""
    return tokenizer.apply_chat_template(
        build_messages(input_text) + [{"role": "assistant", "content": target_text}],
        tokenize=False)


def clean_completion(generated: str) -> str:
    """Cut generated text at the first end token."""
    cut = len(generated)
    for tok in END_TOKENS:
        i = generated.find(tok)
        if i != -1:
            cut = min(cut, i)
    return generated[:cut]


def load_tokenizer(model_id: str = MODEL_ID):
    """Load the tokenizer from the local HF cache only (never downloads)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
