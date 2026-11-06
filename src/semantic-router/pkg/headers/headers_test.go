package headers

import (
	"testing"
)

func TestKVTransferStatusValues(t *testing.T) {
	if KVTransferStatusApplied != "applied" {
		t.Errorf("KVTransferStatusApplied = %q, want %q", KVTransferStatusApplied, "applied")
	}
	if KVTransferStatusFallbackReprefill != "fallback_reprefill" {
		t.Errorf("KVTransferStatusFallbackReprefill = %q, want %q", KVTransferStatusFallbackReprefill, "fallback_reprefill")
	}
	if KVTransferStatusUnsupported != "unsupported" {
		t.Errorf("KVTransferStatusUnsupported = %q, want %q", KVTransferStatusUnsupported, "unsupported")
	}
}

func TestResponseWarningsCodes(t *testing.T) {
	// The v0.4 contract consolidates the hallucination / unverified-factual /
	// response-jailbreak warnings into x-vsr-response-warnings; these are the
	// codes carried in its value. Assert the constants directly — a map keyed by
	// the constant would compare each value to itself and never fail.
	if ResponseWarningHallucination != "hallucination" {
		t.Errorf("ResponseWarningHallucination = %q, want %q", ResponseWarningHallucination, "hallucination")
	}
	if ResponseWarningUnverifiedFactual != "unverified_factual" {
		t.Errorf("ResponseWarningUnverifiedFactual = %q, want %q", ResponseWarningUnverifiedFactual, "unverified_factual")
	}
	if ResponseWarningJailbreak != "response_jailbreak" {
		t.Errorf("ResponseWarningJailbreak = %q, want %q", ResponseWarningJailbreak, "response_jailbreak")
	}
}
