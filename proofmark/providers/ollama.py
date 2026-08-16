"""Ollama provider over stdlib HTTP.

Ollama accepts a JSON Schema in its ``format`` field, which proofmark maps to the
native tier. Older builds accept only ``format: "json"``; that maps to json mode.
"""

import json
from dataclasses import dataclass, field
from urllib import error, request

from proofmark.errors import ProviderError, TierUnsupported
from proofmark.providers.base import Generation, schema_instruction
from proofmark.result import Tier


@dataclass
class OllamaProvider:
    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    name: str = field(default="ollama", init=False)
    supported_tiers: tuple[Tier, ...] = (Tier.NATIVE_STRUCTURED, Tier.JSON_MODE, Tier.TEXT_JSON)

    def generate(self, prompt: str, schema: dict, schema_name: str, tier: Tier) -> Generation:
        body: dict = {"model": self.model, "prompt": prompt, "stream": False}

        if tier is Tier.NATIVE_STRUCTURED:
            body["format"] = schema
        elif tier is Tier.JSON_MODE:
            body["format"] = "json"
            body["prompt"] = prompt + schema_instruction(schema, schema_name)
        else:
            body["prompt"] = prompt + schema_instruction(schema, schema_name)

        req = request.Request(
            url=f"{self.base_url.rstrip('/')}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = self._read_error(exc)
            debug = {"provider": self.name, "model": self.model, "tier": tier.value, "http_status": exc.code}
            if exc.code == 400 and tier is Tier.NATIVE_STRUCTURED:
                raise TierUnsupported(tier.value, detail[:300], debug) from exc
            raise ProviderError(f"HTTP {exc.code}: {detail[:300]}", debug) from exc
        except error.URLError as exc:
            raise ProviderError(
                f"connection failed: {exc.reason}. Is ollama running at {self.base_url}?",
                {"provider": self.name, "model": self.model, "tier": tier.value},
            ) from exc

        return Generation(
            text=str(payload.get("response", "")),
            tier=tier,
            provider=self.name,
            model=self.model,
            incomplete=payload.get("done_reason") == "length",
            debug={
                "provider": self.name,
                "model": self.model,
                "tier": tier.value,
                "done_reason": payload.get("done_reason"),
            },
        )

    def _read_error(self, exc: error.HTTPError) -> str:
        try:
            return exc.read().decode("utf-8")
        except Exception:
            return str(exc.reason)
