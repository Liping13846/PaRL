"""Crawler / Selector 双角色 LLM，支持共享单模型或 PaSa 双 checkpoint。"""

from __future__ import annotations

from typing import Any

from .llm import LLMClient


class DualAgent:
    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.model_cfg = model_cfg
        # shared: 单模型；dual: Crawler+Selector 并行加载；sequential: 轮流加载省显存
        self.load_mode = model_cfg.get("load_mode", "shared")
        self._crawler: LLMClient | None = None
        self._selector: LLMClient | None = None
        self._shared: LLMClient | None = None

    def _uses_dual_checkpoints(self) -> bool:
        return bool(self.model_cfg.get("crawler_name") and self.model_cfg.get("selector_name"))

    def _make_client(self, role: str) -> LLMClient:
        cfg = dict(self.model_cfg)
        role_key = f"{role}_name"
        if role_key in cfg:
            cfg["name"] = cfg[role_key]
        if self.model_cfg.get("use_pasa_prompts"):
            cfg["chat_style"] = "pasa"
        if role == "selector" and self.model_cfg.get("use_pasa_selector_scoring"):
            cfg["use_pasa_selector_scoring"] = True
        return LLMClient(cfg)

    def _get_shared(self) -> LLMClient:
        if self._shared is None:
            self._shared = LLMClient(self.model_cfg)
        return self._shared

    @property
    def crawler(self) -> LLMClient:
        if not self._uses_dual_checkpoints() or self.load_mode == "shared":
            return self._get_shared()
        if self.load_mode == "sequential" and self._selector is not None:
            self._selector.unload()
            self._selector = None
        if self._crawler is None:
            self._crawler = self._make_client("crawler")
        return self._crawler

    @property
    def selector(self) -> LLMClient:
        if not self._uses_dual_checkpoints() or self.load_mode == "shared":
            return self._get_shared()
        if self.load_mode == "sequential" and self._crawler is not None:
            self._crawler.unload()
            self._crawler = None
        if self._selector is None:
            self._selector = self._make_client("selector")
        return self._selector

    @property
    def llm_call_count(self) -> int:
        total = 0
        for client in (self._shared, self._crawler, self._selector):
            if client is not None:
                total += client.llm_call_count
        return total

    def unload_all(self) -> None:
        for client in (self._shared, self._crawler, self._selector):
            if client is not None:
                client.unload()
        self._shared = None
        self._crawler = None
        self._selector = None
