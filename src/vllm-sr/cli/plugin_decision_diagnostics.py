"""Decision diagnostics bounds shared with the Go plugin contract."""

from pydantic import BaseModel, Field, StrictBool


class DecisionDiagnosticsPluginConfig(BaseModel):
    enabled: StrictBool = False
    # Explicit zero selects the Go runtime default.
    max_signals: int = Field(default=32, ge=0, le=128, strict=True)
    max_projections: int = Field(default=16, ge=0, le=64, strict=True)
    max_text_runes: int = Field(default=128, ge=0, le=512, strict=True)
    max_payload_bytes: int = Field(default=16384, ge=0, le=65536, strict=True)
