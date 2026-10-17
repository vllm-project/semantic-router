package llmprotocol

import (
	"errors"
	"testing"
)

func TestValidateUsagePreservesPartialCacheEvidence(t *testing.T) {
	usage := Usage{State: UsageAvailable, InputTotal: authoritativeTestCount(10), InputCacheRead: authoritativeTestCount(3)}
	if err := ValidateUsage(usage); err != nil {
		t.Fatalf("unknown cache-write and uncached buckets were treated as zero: %v", err)
	}
	usage.InputCacheWrite = authoritativeTestCount(4)
	if err := ValidateUsage(usage); err != nil {
		t.Fatalf("valid known subtotal was rejected: %v", err)
	}
	usage.InputCacheWrite = authoritativeTestCount(8)
	if err := ValidateUsage(usage); err == nil {
		t.Fatal("known cache subtotal exceeding input total was accepted")
	}
	usage.InputCacheRead = authoritativeTestCount(11)
	usage.InputCacheWrite = TokenCount{}
	if err := ValidateUsage(usage); err == nil {
		t.Fatal("single cache bucket exceeding input total was accepted")
	}
}

func TestValidateUsageTieredCacheWrites(t *testing.T) {
	usage := Usage{
		State:             UsageAvailable,
		InputTotal:        authoritativeTestCount(10),
		InputCacheWrite:   authoritativeTestCount(6),
		InputCacheWrite5m: authoritativeTestCount(2),
		InputCacheWrite1h: authoritativeTestCount(4),
	}
	if err := ValidateUsage(usage); err != nil {
		t.Fatalf("valid tiered cache writes rejected: %v", err)
	}

	// Mismatched tiers (2 + 3 != 6) rejected with usage_total_mismatch
	usage.InputCacheWrite1h = authoritativeTestCount(3)
	err := ValidateUsage(usage)
	if err == nil {
		t.Fatal("mismatched tier counts accepted")
	}
	var protocolErr *ProtocolError
	if !errors.As(err, &protocolErr) || protocolErr.Code != "usage_total_mismatch" {
		t.Fatalf("expected usage_total_mismatch, got: %v", err)
	}

	// Unknown tier treated as partial/incomplete
	usage.InputCacheWrite1h = TokenCount{}
	if err := ValidateUsage(usage); err != nil {
		t.Fatalf("partial tier count (2 <= 6) rejected: %v", err)
	}

	// Known tier exceeding total (7 > 6) rejected
	usage.InputCacheWrite5m = authoritativeTestCount(7)
	if err := ValidateUsage(usage); err == nil {
		t.Fatal("tier exceeding aggregate cache write accepted")
	}
}
