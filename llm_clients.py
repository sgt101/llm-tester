"""
LLM client wrappers for Anthropic, OpenAI, and Google Gemini.

Each client exposes a single unified method:

    response = client.analyze_image(image_path, prompt)

API keys are read from environment variables by default:
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    GEMINI_API_KEY

Install dependencies:
    uv add anthropic openai google-genai
"""

import base64
import mimetypes
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared response type
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    provider: str
    model: str
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Common interface for all LLM provider clients."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or self.default_model
        self.api_key = api_key or os.environ.get(self.api_key_env, "")
        if not self.api_key:
            raise ValueError(
                f"No API key provided for {self.provider_name}. "
                f"Set the {self.api_key_env} environment variable or pass api_key=."
            )

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def default_model(self) -> str: ...

    @property
    @abstractmethod
    def api_key_env(self) -> str: ...

    @abstractmethod
    def analyze_image(self, image_path: str | Path, prompt: str) -> LLMResponse:
        """Send an image file and a text prompt; return the model's response."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_image_b64(image_path: str | Path) -> tuple[str, str]:
        """Return (base64_string, mime_type) for the given image file."""
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            # Fall back based on suffix
            suffix = path.suffix.lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(suffix, "image/png")
        with open(path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        return b64, mime_type


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    provider_name = "anthropic"
    default_model = "claude-opus-4-6"
    api_key_env = "ANTHROPIC_API_KEY"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        super().__init__(model, api_key)
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("anthropic package is required: uv add anthropic")
        self._client = _anthropic.Anthropic(api_key=self.api_key)

    def analyze_image(self, image_path: str | Path, prompt: str) -> LLMResponse:
        b64, mime_type = self._read_image_b64(image_path)
        message = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text = message.content[0].text if message.content else ""
        return LLMResponse(
            provider=self.provider_name,
            model=self.model,
            text=text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            raw=message,
        )


# ---------------------------------------------------------------------------
# OpenAI (GPT)
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    provider_name = "openai"
    default_model = "gpt-4o"
    api_key_env = "OPENAI_API_KEY"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        super().__init__(model, api_key)
        try:
            from openai import OpenAI as _OpenAI
        except ImportError:
            raise ImportError("openai package is required: uv add openai")
        self._client = _OpenAI(api_key=self.api_key)

    def analyze_image(self, image_path: str | Path, prompt: str) -> LLMResponse:
        b64, mime_type = self._read_image_b64(image_path)
        data_url = f"data:{mime_type};base64,{b64}"
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_completion_tokens=1024,
        )
        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = completion.usage
        return LLMResponse(
            provider=self.provider_name,
            model=self.model,
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw=completion,
        )


# ---------------------------------------------------------------------------
# Google (Gemini)
# ---------------------------------------------------------------------------

class GoogleClient(LLMClient):
    provider_name = "google"
    default_model = "gemini-2.0-flash"
    api_key_env = "GEMINI_API_KEY"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        super().__init__(model, api_key)
        try:
            from google import genai as _genai
            from google.genai import types as _types
        except ImportError:
            raise ImportError("google-genai package is required: uv add google-genai")
        self._genai = _genai
        self._types = _types
        self._client = _genai.Client(api_key=self.api_key)

    def analyze_image(self, image_path: str | Path, prompt: str) -> LLMResponse:
        b64, mime_type = self._read_image_b64(image_path)
        image_part = self._types.Part.from_bytes(
            data=base64.standard_b64decode(b64),
            mime_type=mime_type,
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=[image_part, prompt],
        )
        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            provider=self.provider_name,
            model=self.model,
            text=text,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            raw=response,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "google": GoogleClient,
}


def get_client(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """Return an LLMClient for the named provider.

    Args:
        provider: One of "anthropic", "openai", or "google".
        model:    Model name override. Uses each provider's default when omitted.
        api_key:  API key override. Reads from the environment when omitted.

    Example:
        client = get_client("anthropic")
        response = client.analyze_image("output/png_10_4/composite_0001.png",
                                        "How many elephants are in this image?")
        print(response.text)
    """
    key = provider.lower().strip()
    cls = _PROVIDERS.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {', '.join(_PROVIDERS)}"
        )
    return cls(model=model, api_key=api_key)
