package config

import "testing"

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
