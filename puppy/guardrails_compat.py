"""
Compatibility shim: custom ValidChoices validator for guardrails.

Reflection uses Guard.for_pydantic + messages=... (guardrails 0.6+).  ValidChoices
is reimplemented because guardrails.validators.ValidChoices was deprecated.
"""
from typing import Any, Dict

import guardrails as gd
from guardrails.validator_base import Validator, PassResult, FailResult


# ---------------------------------------------------------------------------
# ValidChoices — custom validator compatible with guardrails 0.4.x
# ---------------------------------------------------------------------------

@gd.register_validator(name="valid_choices_compat", data_type=["integer", "string", "float"])
class ValidChoices(Validator):
    """Validates that a value is one of the allowed choices."""

    def __init__(self, choices=None, on_fail="reask", **kwargs):
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

