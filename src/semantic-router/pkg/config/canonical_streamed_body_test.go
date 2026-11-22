package config

import (
	"strings"
	"testing"
)

// An enabled streamed body accumulates until max_bytes names a ceiling, so
// enabled without one is refused at load.
func TestStreamedBodyNeedsAMaxBytesWhenEnabled(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`version: v0.5
global:
  router:
    streamed_body:
      enabled: true
`))
	if err == nil || !strings.Contains(err.Error(), "max_bytes") {
		t.Fatalf("err = %v, want the enabled streamed body needing max_bytes", err)
	}

	cfg, err := ParseYAMLBytes([]byte(`version: v0.5
global:
  router:
    streamed_body:
      enabled: true
      max_bytes: 10485760
`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MaxStreamedBodyBytes != 10485760 {
		t.Fatalf("max_streamed_body_bytes = %d", cfg.MaxStreamedBodyBytes)
	}

	if _, err := ParseYAMLBytes([]byte(`version: v0.5
global:
  router:
    streamed_body:
      enabled: false
`)); err != nil {
		t.Fatalf("disabled streamed body: %v", err)
	}
}
