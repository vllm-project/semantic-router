package tasks

// NLILabel describes a premise/hypothesis relation independently of an engine.
type NLILabel int

const (
	NLIEntailment NLILabel = iota
	NLINeutral
	NLIContradiction
	NLIUnknown
	NLIError NLILabel = -1
)

func (l NLILabel) String() string {
	switch l {
	case NLIEntailment:
		return "ENTAILMENT"
	case NLINeutral:
		return "NEUTRAL"
	case NLIContradiction:
		return "CONTRADICTION"
	case NLIUnknown:
		return "UNKNOWN"
	default:
		return "ERROR"
	}
}
