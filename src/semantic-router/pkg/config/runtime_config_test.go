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

// LooperConfig.GetMaxResponseBytes bounds a single upstream model response so
// an oversized body cannot exhaust router memory. It mirrors the existing
// GetGRPCMaxMsgSize pattern: an MB knob with a safe default when unset.

func TestLooperConfigGetMaxResponseBytes_DefaultWhenUnset(t *testing.T) {
	cfg := &LooperConfig{}

	if got := cfg.GetMaxResponseBytes(); got != DefaultMaxResponseBytes {
		t.Errorf("GetMaxResponseBytes() = %d, want default %d", got, DefaultMaxResponseBytes)
	}
}

func TestLooperConfigGetMaxResponseBytes_ExplicitMB(t *testing.T) {
	cfg := &LooperConfig{MaxResponseBytesMB: 4}

	want := int64(4) * 1024 * 1024
	if got := cfg.GetMaxResponseBytes(); got != want {
		t.Errorf("GetMaxResponseBytes() = %d, want %d", got, want)
	}
}

func TestLooperConfigGetMaxResponseBytes_NonPositiveFallsBackToDefault(t *testing.T) {
	cfg := &LooperConfig{MaxResponseBytesMB: -5}

	if got := cfg.GetMaxResponseBytes(); got != DefaultMaxResponseBytes {
		t.Errorf("GetMaxResponseBytes() = %d, want default %d for non-positive input", got, DefaultMaxResponseBytes)
	}
}
