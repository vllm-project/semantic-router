package config

import (
	"strconv"
	"strings"
	"testing"
	"time"
)

const drainTimeoutDoc = `
version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  models:
    - name: model-a
      provider_model_id: model-a
      api_format: openai
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
recipes:
  - name: default
`

// parseDrainTimeoutDoc loads a one-listener canonical config whose
// global.router holds router, when it is set.
func parseDrainTimeoutDoc(router string) (*RouterConfig, error) {
	doc := drainTimeoutDoc
	if router != "" {
		doc += "global:\n  router:\n    " + router + "\n"
	}
	return ParseYAMLBytes([]byte(doc))
}

func TestDrainTimeoutStays25SecondsWhenOmitted(t *testing.T) {
	cfg, err := parseDrainTimeoutDoc("")
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.DrainTimeout(); got != 25*time.Second {
		t.Fatalf("DrainTimeout() = %v without drain_timeout_seconds, want 25s", got)
	}
	if got := (*RouterConfig)(nil).DrainTimeout(); got != 25*time.Second {
		t.Fatalf("a nil configuration drains for %v, want 25s", got)
	}
	if exported := CanonicalGlobalFromRouterConfig(cfg).Router.DrainTimeoutSeconds; exported != nil {
		t.Fatalf("an omitted drain timeout exports as %d, want it omitted", *exported)
	}
	if err = ValidateGatewayCapabilities(cfg, GatewayExtProc); err != nil {
		t.Fatalf("an omitted drain timeout is refused behind Envoy: %v", err)
	}
}

// An explicit value is that many seconds, 0 meaning no drain, and exports as
// written; behind Envoy any value is refused, 0 included.
func TestDrainTimeoutSecondsSetsTheStandaloneDrain(t *testing.T) {
	for _, test := range []struct {
		seconds int
		drain   time.Duration
	}{{180, 180 * time.Second}, {0, 0}} {
		cfg, err := parseDrainTimeoutDoc("drain_timeout_seconds: " + strconv.Itoa(test.seconds))
		if err != nil {
			t.Fatal(err)
		}
		if got := cfg.DrainTimeout(); got != test.drain {
			t.Fatalf("drain_timeout_seconds: %d: DrainTimeout() = %v, want %v", test.seconds, got, test.drain)
		}
		if exported := CanonicalGlobalFromRouterConfig(cfg).Router.DrainTimeoutSeconds; exported == nil || *exported != test.seconds {
			t.Fatalf("drain_timeout_seconds: %d exports as %v", test.seconds, exported)
		}
		if err = ValidateGatewayCapabilities(cfg, GatewayStandalone); err != nil {
			t.Fatalf("standalone mode refuses drain_timeout_seconds: %d: %v", test.seconds, err)
		}
		err = ValidateGatewayCapabilities(cfg, GatewayExtProc)
		if err == nil || !strings.Contains(err.Error(), "global.router.drain_timeout_seconds is honored only in standalone mode") {
			t.Fatalf("behind Envoy, which drains the client connections itself, drain_timeout_seconds: %d: err = %v", test.seconds, err)
		}
	}
}

func TestDrainTimeoutSecondsIsBounded(t *testing.T) {
	for _, value := range []string{"-1", "86401"} {
		_, err := parseDrainTimeoutDoc("drain_timeout_seconds: " + value)
		if err == nil || !strings.Contains(err.Error(), "global.router.drain_timeout_seconds must be between 0 and 86400, got "+value) {
			t.Fatalf("drain_timeout_seconds: %s: err = %v", value, err)
		}
	}
	for _, value := range []string{"0", "86400"} {
		if _, err := parseDrainTimeoutDoc("drain_timeout_seconds: " + value); err != nil {
			t.Fatalf("drain_timeout_seconds: %s: %v", value, err)
		}
	}
}
