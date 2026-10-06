package llmprotocol

import "testing"

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
