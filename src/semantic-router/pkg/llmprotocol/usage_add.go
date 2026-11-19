package llmprotocol

import (
	"fmt"
	"math"
)

// AddUsage is the usage of two calls that answered one request: each count
// is the sum of both, authoritative only when both are, and unknown when
// either call did not report it.
func AddUsage(left, right Usage) (Usage, error) {
	if left.State != UsageAvailable || right.State != UsageAvailable {
		return Usage{State: UsageUnavailable}, nil
	}
	result := Usage{State: UsageAvailable}
	pairs := []struct {
		left, right TokenCount
		target      *TokenCount
	}{
		{left.InputUncached, right.InputUncached, &result.InputUncached},
		{left.InputCacheRead, right.InputCacheRead, &result.InputCacheRead},
		{left.InputCacheWrite, right.InputCacheWrite, &result.InputCacheWrite},
		{left.InputCacheWrite5m, right.InputCacheWrite5m, &result.InputCacheWrite5m},
		{left.InputCacheWrite1h, right.InputCacheWrite1h, &result.InputCacheWrite1h},
		{left.OutputReasoning, right.OutputReasoning, &result.OutputReasoning},
		{left.OutputOther, right.OutputOther, &result.OutputOther},
		{left.InputTotal, right.InputTotal, &result.InputTotal},
		{left.OutputTotal, right.OutputTotal, &result.OutputTotal},
		{left.Total, right.Total, &result.Total},
	}
	for _, pair := range pairs {
		sum, err := addTokenCount(pair.left, pair.right)
		if err != nil {
			return Usage{}, err
		}
		*pair.target = sum
	}
	return result, nil
}

func addTokenCount(left, right TokenCount) (TokenCount, error) {
	if left.Value == nil || right.Value == nil {
		return TokenCount{Provenance: UsageUnknown}, nil
	}
	if *left.Value > math.MaxInt64-*right.Value {
		return TokenCount{}, fmt.Errorf("usage of two calls overflows")
	}
	value := *left.Value + *right.Value
	provenance := UsageDerived
	if left.Provenance == UsageAuthoritative && right.Provenance == UsageAuthoritative {
		provenance = UsageAuthoritative
	}
	return TokenCount{Value: &value, Provenance: provenance}, nil
}
