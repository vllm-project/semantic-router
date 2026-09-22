package agenticfacts

import (
	"sort"
	"time"
)

// acceptedScalars holds every validated scalar fact. It is split out from
// Accepted so that it stays a comparable struct: Go allows == on a struct only
// when all its fields are comparable, and slices are not. That lets IsEmpty
// test all sixteen scalars with one comparison against the zero value instead
// of a chain that would need editing every time a field is added.
//
// Each optional numeric fact carries an XxxKnown companion rather than being a
// pointer, matching AgenticSessionContext in pkg/selection. The flag is what
// separates "the caller declared zero" from "the caller declared nothing".
type acceptedScalars struct {
	RootInvocationID     string
	ParentInvocationID   string
	Depth                int
	DelegatedRole        string
	TaskPhase            string
	RemainingTokens      int64
	RemainingTokensKnown bool
	RemainingTimeMs      int64
	RemainingTimeMsKnown bool
	RemainingCost        float64
	RemainingCostKnown   bool
	ContextPortability   string
	Tenant               string
	Residency            string
	TrustLabel           string
	ExpiresAt            time.Time
	ExpiresAtKnown       bool
}

// Accepted holds the facts that survived validation, normalized. It is the only
// shape the rest of the Router consumes: reaching for an Envelope instead would
// bypass the trust boundary this package exists to enforce.
//
// RequiredCapabilities is deduplicated and sorted so that equivalent envelopes
// produce byte-identical Replay records regardless of the order the caller
// listed entries in.
type Accepted struct {
	acceptedScalars
	RequiredCapabilities []string
}

// IsEmpty reports whether these facts can influence selection at all. A nil
// receiver is empty, so callers may test a Result's Accepted pointer without a
// nil check of their own.
func (a *Accepted) IsEmpty() bool {
	if a == nil {
		return true
	}
	return a.acceptedScalars == acceptedScalars{} &&
		len(a.RequiredCapabilities) == 0
}

// dedupSorted returns values with duplicates removed and the remainder sorted,
// so that a list means the same thing however the caller ordered or repeated
// it. Empty input returns nil rather than an empty slice, keeping the JSON
// encoding of an absent list stable.
//
// It deliberately performs no trimming or case folding. Capabilities fold to
// lower case and model names must not, so normalization belongs at the call
// site in validate.go.
//
// Callers must check cardinality bounds against the raw input before calling
// this: deduplicating first would let forty copies of one entry pass a
// sixteen-entry cap.
func dedupSorted(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))

	for _, v := range values {
		if _, ok := seen[v]; ok {
			continue
		}
		seen[v] = struct{}{}
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}
