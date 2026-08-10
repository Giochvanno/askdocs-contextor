"""
config.py — single source of truth for app configuration.

All settings live here. Secrets (the API key) are read from environment
variables or a .env file and are NEVER hardcoded in the source.

API key loading order:
  1) ANTHROPIC_API_KEY environment variable (if already set in the system);
  2) a .env file in the project root (see .env.example).
"""

import os
from dataclasses import dataclass, field

# .env is loaded automatically if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # without python-dotenv we just read system env vars
    pass


@dataclass(frozen=True)
class ModelConfig:
    id: str
    label: str
    price_in: float   # USD per million input tokens
    price_out: float  # USD per million output tokens


# Available models (strings valid at the time of writing).
MODELS: dict[str, ModelConfig] = {
    "haiku": ModelConfig("claude-haiku-4-5", "Haiku — быстро и дёшево", 1.0, 5.0),
    "sonnet": ModelConfig("claude-sonnet-4-6", "Sonnet — умнее, дороже", 3.0, 15.0),
}


@dataclass(frozen=True)
class Settings:
    # --- secrets ---
    api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))

    # --- default model ---
    default_model: str = os.getenv("DEFAULT_MODEL", "haiku")

    # --- generation parameters ---
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))

    # --- context logic ---
    full_mode_token_limit: int = int(os.getenv("FULL_MODE_TOKEN_LIMIT", "30000"))
    retrieval_char_budget: int = int(os.getenv("RETRIEVAL_CHAR_BUDGET", "48000"))

    # --- document chunking ---
    chunk_chars: int = int(os.getenv("CHUNK_CHARS", "2500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "250"))

    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def model(self, key: str | None = None) -> ModelConfig:
        return MODELS.get(key or self.default_model, MODELS["haiku"])


# a single settings instance shared across the whole app
settings = Settings()