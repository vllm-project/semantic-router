// Package ir contains protocol-neutral intermediate-representation extension
// types. The IR core remains *openai.ChatCompletionNewParams (the existing
// router contract); IRExtensions is an additive sidecar populated by inbound
// parsers for fields that do not have an OpenAI-native home.
//
// Nil is the canonical "absent" value: an OpenAI inbound request leaves
// IRExtensions nil end-to-end, and every accessor on *IRExtensions is
// nil-safe.
package ir

// WarningSeverity classifies translation observations for observability.
// Lossy warnings indicate data was discarded or coerced; Info warnings are
// non-actionable; Error warnings indicate translation could not preserve a
// required field.
type WarningSeverity int

const (
	// WarningSeverityInfo is non-actionable context, surfaced for diagnostics only.
	WarningSeverityInfo WarningSeverity = iota
	// WarningSeverityLossy indicates a field was dropped or coerced during translation.
	WarningSeverityLossy
	// WarningSeverityError indicates a required field could not be preserved.
	WarningSeverityError
)

// Warning surfaces a single translation observation. Warnings flow through
// the response header pipeline (X-VSR-Translation-Warning, future PR),
// structured logging, and Prometheus counters.
//
// Field is a dotted JSON path into the inbound body (e.g.
// "messages[3].content[1].cache_control.ttl") so post-hoc analysis can map
// the warning back to the originating client payload.
type Warning struct {
	Field    string
	Reason   WarningReason
	Severity WarningSeverity
	Detail   string
}

// String returns the lowercase token representation of the severity used
// in the response header and metric label.
func (s WarningSeverity) String() string {
	switch s {
	case WarningSeverityInfo:
		return "info"
	case WarningSeverityLossy:
		return "lossy"
	case WarningSeverityError:
		return "error"
	default:
		return "unknown"
	}
}
