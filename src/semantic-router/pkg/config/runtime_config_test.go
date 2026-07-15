package config

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// TestLooperConfigValidateRejectsReservedHeaders proves an operator cannot pin
// a reserved internal header via looper.headers, which would let a configured
// value spoof the internal path or a caller identity.
func TestLooperConfigValidateRejectsReservedHeaders(t *testing.T) {
	for _, reserved := range headers.ReservedInternalHeaders {
		cfg := &LooperConfig{Headers: map[string]string{reserved: "true"}}
		if err := cfg.Validate(); err == nil {
			t.Errorf("Validate() accepted reserved header %q, want error", reserved)
		}
	}
}

// TestLooperConfigValidateRejectsReservedHeaderMixedCase proves the rejection is
// case-insensitive.
func TestLooperConfigValidateRejectsReservedHeaderMixedCase(t *testing.T) {
	cfg := &LooperConfig{Headers: map[string]string{"X-VSR-Inbound-Authorization": "Bearer x"}}
	if err := cfg.Validate(); err == nil {
		t.Fatal("Validate() accepted mixed-case reserved header, want error")
	}
}

// TestLooperConfigValidateAllowsBenignHeaders proves ordinary transport headers
// are still permitted.
func TestLooperConfigValidateAllowsBenignHeaders(t *testing.T) {
	cfg := &LooperConfig{Headers: map[string]string{"X-Trace-Id": "abc", "User-Agent": "vsr"}}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() rejected benign headers: %v", err)
	}
	if err := (&LooperConfig{}).Validate(); err != nil {
		t.Fatalf("Validate() rejected empty headers: %v", err)
	}
}
