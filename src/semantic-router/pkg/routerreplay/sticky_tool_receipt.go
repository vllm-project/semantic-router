package routerreplay

import (
	"strconv"
	"time"
)

// StickyToolSelectionReceipt records one sticky tool-set selection (issue
// #3347). It is content-free by construction: a bounded outcome, a
// closed-vocabulary reason, and counts. It has no field for tool names,
// schemas, prompts, arguments, results, credentials, storage keys,
// fingerprints, or principal and session identifiers.
type StickyToolSelectionReceipt struct {
	Outcome  string `json:"outcome"`
	Reason   string `json:"reason,omitempty"`
	Selected int    `json:"selected"`
	Reused   int    `json:"reused"`
	Added    int    `json:"added"`
	Pinned   int    `json:"pinned"`
	Removed  int    `json:"removed"`
}

// StickyToolSelectionTarget names the Replay outcome target for receipts.
const StickyToolSelectionTarget = "sticky_tool_selection"

// ReplayOutcome renders the receipt as a Replay outcome carrying the same
// bounded fields and nothing else.
func (r StickyToolSelectionReceipt) ReplayOutcome(now time.Time) Outcome {
	return Outcome{
		Timestamp: now.UTC(),
		Source:    "tool_selection",
		Target:    StickyToolSelectionTarget,
		Verdict:   r.Outcome,
		Reason:    r.Reason,
		Metadata: map[string]string{
			"selected": strconv.Itoa(r.Selected),
			"reused":   strconv.Itoa(r.Reused),
			"added":    strconv.Itoa(r.Added),
			"pinned":   strconv.Itoa(r.Pinned),
			"removed":  strconv.Itoa(r.Removed),
		},
	}
}
