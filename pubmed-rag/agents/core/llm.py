# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text generation backend.

Two backends, chosen by QUADRA_GENERATOR:
  vertex  (default) Gemini via the Google Gen AI SDK (needs PROJECT_ID + billing).
  hf      a local Hugging Face instruct model via transformers (free, GPU, no API).
          Model id via QUADRA_HF_MODEL (default Qwen/Qwen2.5-7B-Instruct).
"""

import functools
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


def _backend() -> str:
    return os.environ.get("QUADRA_GENERATOR", "vertex").lower()


# --------------------------------------------------------------- local HF backend
@functools.lru_cache(maxsize=1)
def _hf_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.environ.get("QUADRA_HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    logger.info("loading local generator %s", model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return tok, model


def _hf_generate(prompt: str, *, temperature: float, max_new_tokens: int) -> str:
    import torch

    tok, model = _hf_model()
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def _extract_json(raw: str) -> dict[str, Any]:
    """Pull the first {...} block out of a model reply and parse it."""
    match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    logger.warning("local generate_json got unparseable payload raw=%r", raw[:500])
    return {}


# --------------------------------------------------------------- vertex backend
@functools.lru_cache(maxsize=1)
def get_client():
    from google import genai

    from .config import get_settings

    settings = get_settings()
    return genai.Client(
        vertexai=True, project=settings.project_id, location=settings.location
    )


def generate_text(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_output_tokens: int = 8192,
) -> str:
    if _backend() == "hf":
        return _hf_generate(
            prompt, temperature=temperature, max_new_tokens=max_output_tokens
        )

    from google.genai import types

    from .config import get_settings

    settings = get_settings()
    model_id = model or settings.model_id
    logger.info("generate_text model=%s prompt_chars=%d", model_id, len(prompt))
    response = get_client().models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_output_tokens
        ),
    )
    text = response.text or ""
    logger.info(
        "generate_text model=%s served=%s out_chars=%d",
        model_id,
        getattr(response, "model_version", "?"),
        len(text),
    )
    return text


def generate_json(
    prompt: str, *, model: str | None = None, temperature: float = 0.0
) -> dict[str, Any]:
    """Return parsed JSON, or {} if the model emits something unparseable."""
    if _backend() == "hf":
        raw = _hf_generate(
            prompt + "\n\nRespond with ONLY a single valid JSON object.",
            temperature=temperature, max_new_tokens=1024,
        )
        return _extract_json(raw)

    from google.genai import types

    from .config import get_settings

    settings = get_settings()
    model_id = model or settings.scoring_model_id
    response = get_client().models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature, response_mime_type="application/json"
        ),
    )
    raw = response.text or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "generate_json got unparseable payload model=%s raw=%r", model_id, raw[:500]
        )
        return {}
