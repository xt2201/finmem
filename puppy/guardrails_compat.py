"""
Compatibility shim: provides guardrails 0.3.x/0.4.x API on top of guardrails 0.9.x.

Patched items:
- ValidChoices validator (was in guardrails.validators, now must be a hub validator)
- gd.Guard.from_pydantic (renamed to gd.Guard.for_pydantic in 0.5+, with changed signature)
"""
import re
import logging
from typing import List, Any, Dict, Optional, Callable, Type, Union

import guardrails as gd
from guardrails.validator_base import Validator, PassResult, FailResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ValidChoices — compatible validator for guardrails 0.9.x
# ---------------------------------------------------------------------------

@gd.register_validator(name="valid_choices_compat", data_type=["integer", "string", "float"])
class ValidChoices(Validator):
    """Validates that a value is one of the allowed choices (guardrails 0.3.x API)."""

    def __init__(self, choices=None, on_fail="exception", **kwargs):
        super().__init__(on_fail=on_fail, choices=choices, **kwargs)
        if isinstance(choices, str):
            import ast
            try:
                choices = ast.literal_eval(choices)
            except Exception:
                pass
        self.choices = list(choices) if choices is not None else []

    def validate(self, value: Any, metadata: Dict) -> Any:
        if value in self.choices:
            return PassResult()
        return FailResult(
            error_message=(
                f"Value {value!r} is not in the allowed choices: {self.choices}"
            )
        )


# ---------------------------------------------------------------------------
# _GuardCompat — wraps the new Guard to expose the old calling convention
# ---------------------------------------------------------------------------

class _GuardCompat:
    """
    Thin wrapper that lets the old-style code:

        guard = gd.Guard.from_pydantic(output_class=..., prompt=..., num_reasks=1)
        result = guard(endpoint_func, prompt_params={"investment_info": ...})
        guard.history[0].raw_outputs

    work against a guardrails 0.9.x Guard.
    """

    def __init__(self, guard: gd.Guard, prompt: str = "", num_reasks: int = 1):
        self._guard = guard
        self._prompt_template = prompt
        self._num_reasks = num_reasks

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_prompt(self, prompt_params: Optional[Dict]) -> str:
        """
        Replace ``${var}`` template variables and strip guardrails-specific
        placeholders (e.g. ``${gr.complete_json_suffix_v2}``).
        """
        text = self._prompt_template
        for key, value in (prompt_params or {}).items():
            text = text.replace(f"${{{key}}}", str(value))
        # Remove any remaining ${gr.*} placeholders
        text = re.sub(r"\$\{gr\.[^}]+\}", "", text)
        # Clean up stray '}' that can appear after stripping the suffix
        text = re.sub(r"\s*}\s*$", "", text.rstrip())
        return text.strip()

    # ------------------------------------------------------------------
    # __call__ — old-style: guard(endpoint_func, prompt_params=...)
    # ------------------------------------------------------------------

    def __call__(
        self,
        llm_api: Optional[Callable] = None,
        *args,
        prompt_params: Optional[Dict] = None,
        **kwargs,
    ):
        rendered = self._render_prompt(prompt_params)
        messages = [{"role": "user", "content": rendered}]
        return self._guard(
            llm_api,
            *args,
            messages=messages,
            num_reasks=self._num_reasks,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # history — delegate to the inner guard
    # ------------------------------------------------------------------

    @property
    def history(self):
        return self._guard.history


# ---------------------------------------------------------------------------
# Monkey-patch gd.Guard.from_pydantic
# ---------------------------------------------------------------------------

def _from_pydantic_compat(cls, output_class, prompt="", num_reasks=1, **kwargs):
    """Backward-compatible ``Guard.from_pydantic`` shim."""
    inner = gd.Guard.for_pydantic(output_class=output_class)
    return _GuardCompat(guard=inner, prompt=prompt, num_reasks=num_reasks)


# Only patch once
if not hasattr(gd.Guard, "from_pydantic"):
    gd.Guard.from_pydantic = classmethod(_from_pydantic_compat.__func__  # type: ignore[attr-defined]
                                          if hasattr(_from_pydantic_compat, "__func__")
                                          else _from_pydantic_compat)
    # Simpler approach: attach as a plain classmethod
    gd.Guard.from_pydantic = classmethod(lambda cls, **kw: _from_pydantic_compat(cls, **kw))  # type: ignore[assignment]
