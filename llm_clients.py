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

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model or self.default_model
        self.api_key = api_key or os.environ.get(self.api_key_env, "")
        self.temperature = temperature
        self.max_tokens = max_tokens
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

    def __init__(self, model: str | None = None, api_key: str | None = None, temperature: float = 0.0, max_tokens: int = 2048) -> None:
        super().__init__(model, api_key, temperature, max_tokens)
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("anthropic package is required: uv add anthropic")
        self._client = _anthropic.Anthropic(api_key=self.api_key)

    def analyze_image(self, image_path: str | Path, prompt: str) -> LLMResponse:
        b64, mime_type = self._read_image_b64(image_path)
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
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

    def __init__(self, model: str | None = None, api_key: str | None = None, temperature: float = 0.0, max_tokens: int = 2048) -> None:
        super().__init__(model, api_key, temperature, max_tokens)
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
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
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

    def __init__(self, model: str | None = None, api_key: str | None = None, temperature: float = 0.0, max_tokens: int = 2048) -> None:
        super().__init__(model, api_key, temperature, max_tokens)
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
            config=self._types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
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
# MLX-VLM (local, Apple Silicon)
# ---------------------------------------------------------------------------

class MLXClient(LLMClient):
    """Run a vision-language model locally via mlx-vlm (Apple Silicon only).

    The model is a Hugging Face repo ID of an MLX-compatible VLM, e.g.:
        mlx-community/Qwen2-VL-2B-Instruct-4bit
        mlx-community/llava-1.5-7b-4bit

    No API key is required.  Set provider = "mlx" in config.toml and supply
    the repo ID (or local path) as the model name.
    """

    provider_name = "mlx"
    default_model = "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
    api_key_env = ""  # unused

    def __init__(self, model: str | None = None, api_key: str | None = None, temperature: float = 0.0, max_tokens: int = 2048) -> None:
        # Skip the base-class API-key requirement — no key needed for local inference.
        self.model = model or self.default_model
        self.api_key = ""
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from mlx_vlm import load, generate
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError:
            raise ImportError("mlx-vlm package is required: uv add mlx-vlm")

        self._generate = generate
        self._apply_chat_template = apply_chat_template

        print(f"Loading MLX model '{self.model}' …", flush=True)
        self._model, self._processor = load(self.model)
        self._config = load_config(self.model)

    def analyze_image(self, image_path: str | Path, prompt: str) -> LLMResponse:
        image_path = str(Path(image_path).resolve())
        formatted_prompt = self._apply_chat_template(
            self._processor, self._config, prompt, num_images=1
        )
        output = self._generate(
            self._model,
            self._processor,
            formatted_prompt,
            [image_path],
            verbose=False,
            max_tokens=self.max_tokens,
            temp=self.temperature,
        )
        if isinstance(output, str):
            text = output
        elif hasattr(output, "text"):
            text = output.text
        else:
            text = str(output)
        return LLMResponse(
            provider=self.provider_name,
            model=self.model,
            text=text,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "google": GoogleClient,
    "mlx": MLXClient,
}


def get_client(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> LLMClient:
    """Return an LLMClient for the named provider.

    Args:
        provider:    One of "anthropic", "openai", "google", or "mlx".
        model:       Model name override. Uses each provider's default when omitted.
        api_key:     API key override. Reads from the environment when omitted.
                     Not used for the "mlx" provider.
        temperature: Sampling temperature (default 0.0 for deterministic output).
        max_tokens:  Maximum tokens in the response (default 2048).
    """
    key = provider.lower().strip()
    cls = _PROVIDERS.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {', '.join(_PROVIDERS)}"
        )
    return cls(model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
