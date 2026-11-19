package llmprotocol

import (
	"math"
	"testing"
)

func TestAddUsageSumsEveryCountOfTwoCalls(t *testing.T) {
	authoritative := func(value int64) TokenCount { return TokenCount{Value: Int64(value), Provenance: UsageAuthoritative} }
	first := Usage{
		State: UsageAvailable, InputUncached: authoritative(15), InputCacheRead: authoritative(5),
		InputTotal: authoritative(20), OutputReasoning: authoritative(2), OutputOther: authoritative(3),
		OutputTotal: authoritative(5), Total: authoritative(25),
	}
	second := Usage{
		State: UsageAvailable, InputUncached: authoritative(1), InputCacheRead: authoritative(20),
		InputTotal: authoritative(21), OutputReasoning: TokenCount{Value: Int64(0), Provenance: UsageDerived},
		OutputOther: authoritative(7), OutputTotal: authoritative(7), Total: authoritative(28),
	}
	sum, err := AddUsage(first, second)
	if err != nil {
		t.Fatal(err)
	}
	for name, test := range map[string]struct {
		count      TokenCount
		want       int64
		provenance UsageProvenance
	}{
		"input uncached":   {sum.InputUncached, 16, UsageAuthoritative},
		"input cache read": {sum.InputCacheRead, 25, UsageAuthoritative},
		"input total":      {sum.InputTotal, 41, UsageAuthoritative},
		"output reasoning": {sum.OutputReasoning, 2, UsageDerived},
		"output other":     {sum.OutputOther, 10, UsageAuthoritative},
		"output total":     {sum.OutputTotal, 12, UsageAuthoritative},
		"total":            {sum.Total, 53, UsageAuthoritative},
	} {
		if test.count.Value == nil || *test.count.Value != test.want || test.count.Provenance != test.provenance {
			t.Errorf("%s = %+v, want %d %s", name, test.count, test.want, test.provenance)
		}
	}
	if sum.State != UsageAvailable || sum.InputCacheWrite.Value != nil || sum.InputCacheWrite.Provenance != UsageUnknown {
		t.Fatalf("a count neither call reported must stay unknown: %+v", sum)
	}
}

func TestAddUsageIsUnknownWhenEitherCallsUsageIs(t *testing.T) {
	known := Usage{State: UsageAvailable, InputTotal: TokenCount{Value: Int64(1), Provenance: UsageAuthoritative}}
	for name, pair := range map[string][2]Usage{
		"first unknown":  {{State: UsageUnavailable}, known},
		"second unknown": {known, {State: UsageUnavailable}},
	} {
		sum, err := AddUsage(pair[0], pair[1])
		if err != nil || sum.State != UsageUnavailable {
			t.Errorf("%s: AddUsage = %+v, %v", name, sum, err)
		}
	}
	huge := Usage{State: UsageAvailable, Total: TokenCount{Value: Int64(math.MaxInt64), Provenance: UsageAuthoritative}}
	if _, err := AddUsage(huge, huge); err == nil {
		t.Fatal("an overflowing sum must fail")
	}
}
