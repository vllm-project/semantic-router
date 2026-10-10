package agenticfacts

import (
	"sort"
)

// Rejection records one field-level validation failure.
type Rejection struct {
	// Field is the JSON path of the offending field, for example
	// "lineage.depth" or "required_capabilities". It is empty for
	// envelope-level failures such as malformed JSON.
	Field string `json:"field,omitempty"`

	// Reason is one of the Reason constants in reasons.go.
	Reason string `json:"reason"`
}

// Result reports the outcome of validating one agentic facts envelope.
type Result struct {
	// Accepted holds the facts that passed validation. It is nil when no
	// envelope was presented and when a presented envelope failed validation;
	// Rejections distinguishes those two cases. Validation is all-or-nothing,
	// so Accepted is non-nil only when Rejections is empty.
	Accepted *Accepted `json:"accepted,omitempty"`

	// Rejections lists every field-level failure in a stable order. A
	// non-empty Rejections means the envelope was dropped and the request
	// routes as if no facts had been presented.
	Rejections []Rejection `json:"rejections,omitempty"`
}

// Rejected reports whether a presented envelope failed validation. The request
// still routes normally; the caller records the reasons as diagnostics.
func (r Result) Rejected() bool {
	return len(r.Rejections) > 0
}

// HasFacts reports whether validation produced facts that can influence
// selection.
func (r Result) HasFacts() bool {
	return !r.Accepted.IsEmpty()
}

// reject appends one validation failure to the result.
func (r *Result) reject(field string, reason string) {
	r.Rejections = append(r.Rejections, Rejection{
		Field:  field,
		Reason: reason,
	})
}

// sortRejections orders rejections so that validating the same bytes twice
// produces an identical Result regardless of the order the checks ran in.
func sortRejections(rejections []Rejection) {

	sort.Slice(rejections, func(i, j int) bool {
		if rejections[i].Field != rejections[j].Field {
			return rejections[i].Field < rejections[j].Field
		}
		return rejections[i].Reason < rejections[j].Reason
	})

}
