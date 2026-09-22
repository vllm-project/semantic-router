// Package agenticfacts validates the bounded, versioned selection-facts
// envelope that external agent runtimes present at the Router's request
// boundary, as specified by Epic #2994 and Feature #3379.
//
// The package separates the wire shape from the validated shape. Envelope is
// the untrusted JSON as presented; Accepted holds only what survived
// validation, normalized. Nothing outside this package should read an Envelope
// directly, so the type system carries the trust boundary.
//
// Validation is bounded in every dimension the proposal names: payload size,
// delegation depth, list cardinality, string length, and lifetime. It is also
// deterministic: the same bytes produce the same Result, with rejections in a
// stable order, so Replay provenance stays reproducible across refactors.
//
// Failure policy is all-or-nothing. A presented envelope that fails any check
// is dropped whole, and the request routes exactly as if no facts had been
// presented; the reasons survive as content-minimized diagnostics. Partial
// acceptance is deliberately not offered, because it would let whoever
// populates the envelope shed an inconvenient constraint by malforming that one
// field while keeping the rest.
//
// Dropping facts is safe because facts only ever narrow within the candidate
// set an operator already configured in decisions[].modelRefs. Falling back
// returns the request to operator policy rather than letting it escape policy.
// The corollary is a standing rule for anything built on this package:
// caller-supplied facts must never be the sole enforcement point for a policy.
// Anything that must hold regardless of caller input belongs in config.
//
// Trust establishment is not done here. This package answers "is this envelope
// well-formed and within bounds", not "did it arrive from an authenticated
// gateway". ReasonUntrusted exists for the ingestion seam that answers the
// second question.
package agenticfacts

// SchemaVersion is the only envelope version this build can interpret. An
// envelope declaring anything else is rejected rather than partially read,
// because every field check below assumes these semantics.
const SchemaVersion = "1"

// Envelope is the untrusted wire shape presented at the request boundary. It
// mirrors the JSON exactly and carries no guarantees: every field is
// caller-controlled until Validate has run. Callers outside this package
// consume Accepted instead.
//
// ExpiresAt is a string rather than a time.Time so that an unparseable
// timestamp becomes a field-level rejection with a reason, instead of an
// unmarshal error that discards the whole payload before any other field can be
// reported.
type Envelope struct {
	Version              string         `json:"version"`
	Lineage              *Lineage       `json:"lineage,omitempty"`
	DelegatedRole        string         `json:"delegated_role,omitempty"`
	TaskPhase            string         `json:"task_phase,omitempty"`
	Budget               *Budget        `json:"budget,omitempty"`
	RequiredCapabilities []string       `json:"required_capabilities,omitempty"`
	ContextPortability   string         `json:"context_portability,omitempty"`
	TrustBoundary        *TrustBoundary `json:"trust_boundary,omitempty"`
	ExpiresAt            string         `json:"expires_at,omitempty"`
}

// Lineage identifies where the current subtask sits in an external runtime's
// delegation graph. The Router uses it for continuity guards, provenance, and
// conflict detection; it does not reconstruct or traverse the graph.
type Lineage struct {
	RootInvocationID   string `json:"root_invocation_id,omitempty"`
	ParentInvocationID string `json:"parent_invocation_id,omitempty"`
	Depth              int    `json:"depth,omitempty"`
}

// Budget carries the caller's remaining allowance. The counters are pointers so
// that an absent budget stays distinguishable from an exhausted one: a
// remaining_tokens of zero is a real, spent budget and must not read as "no
// budget declared".
//
// The proposal names "remaining token, time, or cost counters", so all three
// concepts are evidenced. The exact field names and the millisecond unit are
// chosen here and recorded as open in PL-0042.
type Budget struct {
	RemainingTokens *int64   `json:"remaining_tokens,omitempty"`
	RemainingTimeMs *int64   `json:"remaining_time_ms,omitempty"`
	RemainingCost   *float64 `json:"remaining_cost,omitempty"`
}

// TrustBoundary carries tenant scope, data residency, and a trust label. These
// inform privacy and containment decisions, but they are caller-supplied and so
// must never be the only thing enforcing a residency or authorization policy;
// see the package comment.
type TrustBoundary struct {
	Tenant    string `json:"tenant,omitempty"`
	Residency string `json:"residency,omitempty"`
	Label     string `json:"label,omitempty"`
}
