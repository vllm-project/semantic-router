package sessiontools

import (
	"errors"
	"time"
)

const (
	// MaxSelectionInputs bounds each caller-supplied catalog, ranking, or
	// observation list. It is a library work bound, not a retained-state limit.
	MaxSelectionInputs = 4096
	// MaxSelectionTurn is portable across 32- and 64-bit consumers of
	// ToolState.FirstSeenTurn. Exhaustion fails without resetting continuity.
	MaxSelectionTurn = 1<<31 - 1
)

var (
	ErrInvalidSelection = errors.New("sessiontools: invalid selection input")
	ErrStateTooLarge    = errors.New("sessiontools: state exceeds byte bound")
	ErrTurnExhausted    = errors.New("sessiontools: selection turn exhausted")
)

// ToolIdentity never contains definitions or authorization decisions. The
// caller must supply identities eligible under the current request's policy.
type ToolIdentity struct {
	Name                  string
	DefinitionFingerprint string
}

// RankedTool is an ordinary request-time candidate; Score must be finite.
type RankedTool struct {
	ToolIdentity
	Score float64
}

// SelectionBounds contains resolved configuration, including explicit zero
// growth. A cold selection may seed up to MaxTools; the growth budget applies
// only to subsequent relevance additions. Pins and requirements share the
// hard count/byte limits but do not consume the relevance growth budget.
type SelectionBounds struct {
	MaxTools           int
	MaxNewToolsPerTurn int
	PinCalledTools     bool
	MaxStateBytes      int
}

// Fingerprints describe the current effective policy, catalog, and capability
// set. They are compared in full, independently of any storage-key prefix.
type Fingerprints struct {
	Policy     string
	Catalog    string
	Capability string
}

// SelectionInput is request-local evidence, frozen by Manager.Update before
// touching storage. Eligible is authoritative for this call; Called is only a
// preference and cannot add an ineligible identity. Required names must all
// appear in the result or RequiredToolsError is returned. Neither this layer
// nor its stored state grants authorization.
type SelectionInput struct {
	Eligible     []ToolIdentity
	Ranked       []RankedTool
	Called       []string
	Required     []string
	Bounds       SelectionBounds
	Fingerprints Fingerprints
	StrategyID   string
}

// MergeInput supplies an optional snapshot and an explicit clock to the pure
// planner. Merge never mutates any input. Now and TTL establish proposed
// lifecycle metadata; a Store stamps the actual write's revision and expiry.
type MergeInput struct {
	SelectionInput
	Prior *State
	Now   time.Time
	TTL   time.Duration
}

type (
	SelectionOutcome string
	SelectionReason  string
)

const (
	OutcomeSeeded  SelectionOutcome = "seeded"
	OutcomeReused  SelectionOutcome = "reused"
	OutcomeUpdated SelectionOutcome = "updated"
	OutcomeReset   SelectionOutcome = "reset"

	ReasonNone              SelectionReason = ""
	ReasonMissing           SelectionReason = "missing"
	ReasonExpired           SelectionReason = "expired"
	ReasonInvalidState      SelectionReason = "invalid_state"
	ReasonPolicyChanged     SelectionReason = "policy_changed"
	ReasonCatalogChanged    SelectionReason = "catalog_changed"
	ReasonCapabilityChanged SelectionReason = "capability_changed"
	ReasonClockRegressed    SelectionReason = "clock_regressed"
	ReasonPinCapacity       SelectionReason = "pin_capacity_exceeded"
)

// SelectionReceipt is bounded and content-free. Counts describe the proposed
// merge; Manager returns it only after a confirmed successful CAS. No tool,
// principal, session, storage key, fingerprint, or arbitrary error is included.
type SelectionReceipt struct {
	Outcome  SelectionOutcome
	Reason   SelectionReason
	Selected int
	Reused   int
	Added    int
	Pinned   int
	Removed  int
}

// MergeResult is a proposal, not a stored snapshot: State.Revision is zero.
// Tools and State.Tools have independent backing arrays.
type MergeResult struct {
	Tools   []ToolIdentity
	State   State
	Receipt SelectionReceipt
}

// RequiredToolsError means explicit requirements cannot be honored within
// eligibility or capacity. Runtime adapters must not silently weaken them.
type RequiredToolsError struct {
	Unavailable bool
}

func (e *RequiredToolsError) Error() string {
	if e.Unavailable {
		return "sessiontools: required tool is not currently eligible"
	}
	return "sessiontools: required tools exceed selection capacity"
}
