package ir

import (
	"testing"
)

func TestAppendWarning_NilSafe(t *testing.T) {
	var ext *IRExtensions
	ext.AppendWarning(Warning{Field: "x", Reason: "dropped"})
}

func TestAppendWarning_AppendsToSlice(t *testing.T) {
	ext := &IRExtensions{}
	ext.AppendWarning(Warning{Field: "a", Reason: "dropped", Severity: WarningSeverityLossy})
	ext.AppendWarning(Warning{Field: "b", Reason: "coerced", Severity: WarningSeverityInfo})

	if len(ext.Warnings) != 2 {
		t.Fatalf("expected 2 warnings, got %d", len(ext.Warnings))
	}
	if ext.Warnings[0].Field != "a" || ext.Warnings[0].Severity != WarningSeverityLossy {
		t.Fatalf("first warning unexpected: %+v", ext.Warnings[0])
	}
	if ext.Warnings[1].Field != "b" || ext.Warnings[1].Severity != WarningSeverityInfo {
		t.Fatalf("second warning unexpected: %+v", ext.Warnings[1])
	}
}

func TestSetCacheControl_NilSafe(t *testing.T) {
	var ext *IRExtensions
	ext.SetCacheControl("system.0", CacheControlSpec{Type: "ephemeral", TTL: "5m"})
}

func TestSetCacheControl_LazyInit(t *testing.T) {
	ext := &IRExtensions{}
	ext.SetCacheControl("system.0", CacheControlSpec{Type: "ephemeral", TTL: "5m"})
	ext.SetCacheControl("messages[0].content[1]", CacheControlSpec{Type: "ephemeral", TTL: "1h"})

	if got := ext.CacheControl["system.0"].TTL; got != "5m" {
		t.Fatalf("expected ttl=5m, got %q", got)
	}
	if got := ext.CacheControl["messages[0].content[1]"].TTL; got != "1h" {
		t.Fatalf("expected ttl=1h, got %q", got)
	}
}

func TestSetThinkingSignature_LazyInit(t *testing.T) {
	ext := &IRExtensions{}
	ext.SetThinkingSignature("messages[2].content[0]", "sig-abc")

	if got := ext.ThinkingSignatures["messages[2].content[0]"]; got != "sig-abc" {
		t.Fatalf("expected signature=sig-abc, got %q", got)
	}
}

func TestSetToolStrict_LazyInit(t *testing.T) {
	ext := &IRExtensions{}
	ext.SetToolStrict("search", true)
	ext.SetToolStrict("calc", false)

	if !ext.ToolStrict["search"] {
		t.Fatalf("expected strict=true for search")
	}
	if ext.ToolStrict["calc"] {
		t.Fatalf("expected strict=false for calc")
	}
}

func TestNilExtensions_AccessorsAreNoOps(t *testing.T) {
	var ext *IRExtensions
	ext.SetCacheControl("x", CacheControlSpec{})
	ext.SetThinkingSignature("x", "y")
	ext.SetToolStrict("x", true)
	ext.AppendWarning(Warning{})
}

func TestWarningSeverityValues(t *testing.T) {
	// Stability check: severities are part of the wire contract via metrics
	// labels, so document their integer values.
	if WarningSeverityInfo != 0 {
		t.Fatalf("WarningSeverityInfo must be 0, got %d", WarningSeverityInfo)
	}
	if WarningSeverityLossy != 1 {
		t.Fatalf("WarningSeverityLossy must be 1, got %d", WarningSeverityLossy)
	}
	if WarningSeverityError != 2 {
		t.Fatalf("WarningSeverityError must be 2, got %d", WarningSeverityError)
	}
}
