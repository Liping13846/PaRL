"""LLM 推理封装：支持本地量化模型与 API 两种模式。"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.cfg = model_cfg
        self.mode = model_cfg.get("mode", "local")
        self.max_new_tokens = model_cfg.get("max_new_tokens", 256)
        self._model = None
        self._tokenizer = None
        self.llm_call_count = 0

    def _ensure_local_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.cfg["name"]
        model_path = Path(model_name)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent.parent / model_name
        if model_path.exists():
            model_name = str(model_path)
            logger.info("Using local model path: %s", model_name)

        quantization = self.cfg.get("quantization", "none")
        load_kwargs: dict[str, Any] = {}

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "cpu"
            logger.warning("CUDA unavailable, loading model on CPU (slower inference)")

        if quantization == "int4":
            if not use_cuda:
                logger.warning("INT4 requires CUDA, falling back to float32 on CPU")
                load_kwargs["dtype"] = torch.float32
            else:
                try:
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig

                    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                    )
                    logger.info("Using INT4 NF4 quantization (compute_dtype=%s)", compute_dtype)
                except ImportError:
                    logger.warning("bitsandbytes unavailable, falling back to fp16")
                    load_kwargs["dtype"] = torch.float16
        elif quantization == "none":
            load_kwargs["dtype"] = "auto" if use_cuda else torch.float32
        else:
            load_kwargs["dtype"] = torch.float32

        load_kwargs["low_cpu_mem_usage"] = True

        logger.info("Loading local model: %s (%s)", model_name, quantization)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            local_files_only=Path(model_name).exists(),
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=Path(model_name).exists(),
            **load_kwargs,
        )
        self._model.eval()

    def _model_device(self):
        if hasattr(self._model, "device") and getattr(self._model, "device", None) is not None:
            device = self._model.device
            if str(device) != "meta":
                return device
        return next(self._model.parameters()).device

    def unload(self) -> None:
        if self._model is None:
            return
        import gc

        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def chat(self, prompt: str, system: str | None = None) -> str:
        self.llm_call_count += 1
        if self.mode == "api":
            return self._chat_api(prompt, system=system)
        self._ensure_local_model()
        return self._chat_local(prompt, system=system)

    def _chat_local(self, prompt: str, system: str | None = None) -> str:
        text = self._format_chat_text(prompt, system=system)
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model_device())
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _format_chat_text(self, prompt: str, system: str | None = None) -> str:
        max_input_len = int(self.cfg.get("max_input_tokens", 2048))
        if self.cfg.get("chat_style") == "pasa":
            return self._tokenizer.apply_chat_template(
                [{"content": prompt.strip(), "role": "user"}],
                tokenize=False,
                max_length=max_input_len,
                add_generation_prompt=True,
            )
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt.strip()})
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _chat_api(self, prompt: str, system: str | None = None) -> str:
        api_key = os.getenv(self.cfg.get("api_key_env", "DEEPSEEK_API_KEY"), "")
        if not api_key:
            raise RuntimeError("API key not set for LLM API mode")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{self.cfg['api_base'].rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.cfg.get("api_model", "deepseek-chat"),
                "messages": messages,
                "max_tokens": self.max_new_tokens,
                "temperature": 0,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def infer_relevance_score(self, prompt: str) -> float:
        """兼容 PaSa Selector 的 True/False 打分方式。"""
        response = self.chat(prompt)
        decision = self._extract_decision(response)
        return 1.0 if decision else 0.0

    def _selector_max_tokens(self) -> int:
        return int(self.cfg.get("selector_max_new_tokens", 64))

    def batch_infer_relevance(self, prompts: list[str]) -> list[float]:
        if not prompts:
            return []
        if self.mode == "api":
            return [self.infer_relevance_score(p) for p in prompts]
        if self.cfg.get("use_pasa_selector_scoring"):
            return self.batch_infer_score(prompts)

        self.llm_call_count += 1
        self._ensure_local_model()
        responses = self._batch_chat_local(prompts, max_new_tokens=self._selector_max_tokens())
        return [1.0 if self._extract_decision(text) else 0.0 for text in responses]

    def batch_infer_score(self, prompts: list[str]) -> list[float]:
        """PaSa Selector：生成 1 token，取 True 的概率作为相关性分数。"""
        if not prompts:
            return []
        self.llm_call_count += 1
        self._ensure_local_model()
        max_input_len = int(self.cfg.get("max_input_tokens", 992))
        texts = [self._format_chat_text(p) for p in prompts]
        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        )
        device = self._model_device()
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)

        outputs = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        true_token_id = self._tokenizer.convert_tokens_to_ids("True")
        if true_token_id is None:
            logger.warning("Tokenizer has no 'True' token, falling back to text parsing")
            responses = self._batch_chat_local(prompts, max_new_tokens=self._selector_max_tokens())
            return [1.0 if self._extract_decision(text) else 0.0 for text in responses]
        probs = outputs.scores[0].softmax(dim=-1)[:, true_token_id].cpu().tolist()
        return [float(p) for p in probs]

    def _batch_chat_local(
        self,
        prompts: list[str],
        system: str | None = None,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        token_budget = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        max_input_len = int(self.cfg.get("max_input_tokens", 2048))

        texts: list[str] = []
        for prompt in prompts:
            texts.append(self._format_chat_text(prompt, system=system))

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        ).to(self._model_device())
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=token_budget,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        responses: list[str] = []
        for idx, in_len in enumerate(input_lengths):
            generated = outputs[idx][int(in_len) :]
            responses.append(self._tokenizer.decode(generated, skip_special_tokens=True).strip())
        return responses

    @staticmethod
    def _extract_decision(text: str) -> bool:
        match = re.search(r"Decision:\s*(True|False)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
        lowered = text.lower()
        if "true" in lowered and "false" not in lowered:
            return True
        return False

    def parse_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        response = self.chat(prompt, system=system)
        return self._extract_json(response)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        candidate = fenced.group(1) if fenced else text
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1:
            return {}
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return {}
