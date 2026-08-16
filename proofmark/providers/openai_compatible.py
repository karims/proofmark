"""OpenAI-compatible chat completions over stdlib HTTP.

No SDK dependency. Works against api.openai.com and against anything exposing the
same ``/chat/completions`` surface -- vLLM, Together, Groq, LM Studio, OpenRouter.
"""

import json
from dataclasses import dataclass, field
from urllib import error, request

from proofmark.errors import ProviderError, TierUnsupported
from proofmark.providers.base import Generation, schema_instruction
from proofmark.result import Tier

_TIER_UNSUPPORTED_MARKERS = (
    "response_format",
    "json_schema",
    "not supported",
    "unsupported",
    "unrecognized request argument",
    "invalid_request_error",
    "does not support",
)


@dataclass
class OpenAICompatibleProvider:
    """Chat-completions provider with per-tier request shaping.

    Tier degradation is driven by the pipeline: this class raises
    :class:`~proofmark.errors.TierUnsupported` when the server rejects a tier's
    request shape, and :class:`~proofmark.errors.ProviderError` for anything else.
    Distinguishing the two is what keeps a bad API key from being silently
    "recovered" into a prompt-and-pray generation.
    """

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 60.0
    max_tokens: int | None = None
    temperature: float | None = None
    name: str = field(default="openai-compatible", init=False)
    supported_tiers: tuple[Tier, ...] = (Tier.NATIVE_STRUCTURED, Tier.JSON_MODE, Tier.TEXT_JSON)

    def generate(self, prompt: str, schema: dict, schema_name: str, tier: Tier) -> Generation:
        body: dict = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            body["temperature"] = self.temperature

        if tier is Tier.NATIVE_STRUCTURED:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            }
        elif tier is Tier.JSON_MODE:
            body["response_format"] = {"type": "json_object"}
            body["messages"][0]["content"] = prompt + schema_instruction(schema, schema_name)
        else:
            body["messages"][0]["content"] = prompt + schema_instruction(schema, schema_name)

        payload = self._post(body, tier)
        return self._to_generation(payload, tier)

    def _post(self, body: dict, tier: Tier) -> dict:
        req = request.Request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = self._read_error(exc)
            debug = {
                "provider": self.name,
                "model": self.model,
                "tier": tier.value,
                "http_status": exc.code,
                "sanitized_message": detail[:500],
            }
            # 400 with a response_format complaint means "this model cannot do this
            # tier" -- degrade. Any other 4xx/5xx is a real failure; degrading would
            # turn an auth error into a mysteriously low-quality result.
            if exc.code == 400 and self._looks_like_tier_rejection(detail):
                raise TierUnsupported(tier.value, detail[:300], debug) from exc
            raise ProviderError(f"HTTP {exc.code}: {detail[:300]}", debug) from exc
        except error.URLError as exc:
            raise ProviderError(
                f"connection failed: {exc.reason}",
                {"provider": self.name, "model": self.model, "tier": tier.value},
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "provider returned a non-JSON envelope",
                {"provider": self.name, "model": self.model, "tier": tier.value},
            ) from exc

    def _looks_like_tier_rejection(self, detail: str) -> bool:
        lowered = detail.lower()
        return any(marker in lowered for marker in _TIER_UNSUPPORTED_MARKERS)

    def _read_error(self, exc: error.HTTPError) -> str:
        try:
            raw = exc.read().decode("utf-8")
        except Exception:
            return exc.reason if isinstance(exc.reason, str) else "unknown error"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(parsed, dict):
            detail = parsed.get("error")
            if isinstance(detail, dict):
                return str(detail.get("message") or detail)
            if detail is not None:
                return str(detail)
        return raw

    def _to_generation(self, payload: dict, tier: Tier) -> Generation:
        choices = payload.get("choices") or []
        if not choices:
            return Generation(
                text=None, tier=tier, provider=self.name, model=self.model, debug={"reason": "no_choices"}
            )

        choice = choices[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")

        debug = {
            "provider": self.name,
            "model": payload.get("model") or self.model,
            "tier": tier.value,
            "finish_reason": finish_reason,
            "usage": payload.get("usage"),
        }

        return Generation(
            text=message.get("content"),
            tier=tier,
            provider=self.name,
            model=self.model,
            refusal=message.get("refusal"),
            incomplete=finish_reason == "length",
            debug=debug,
        )
