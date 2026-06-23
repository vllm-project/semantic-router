// Package sessionbudget computes graduated token-budget enforcement stages from
// a session's cumulative token usage. It is pure logic with no I/O so it can be
// unit-tested in isolation and reused by the extproc dispatch path.
//
// The model implements WRP vision-paper "Open Opportunity 5 (Runtime
// token-budget enforcement for agent sessions)": instead of a binary deny, an
// over-budget session escalates through a graduated ladder
// (shape tools -> compress -> downgrade -> terminate).
package sessionbudget

// Stage is the graduated-response level selected for a session given its
// cumulative token usage relative to its configured budget.
type Stage int

const (
	// StageNone means the session is within budget (or enforcement is disabled);
	// no action is taken.
	StageNone Stage = iota
	// StageShapeTools narrows the tool catalog to the highest-value subset.
	StageShapeTools
	// StageCompress compresses the prompt/history before dispatch.
	StageCompress
	// StageDowngrade routes the next turn to a cheaper model.
	StageDowngrade
	// StageTerminate returns a budget-exceeded signal to the orchestrator.
	StageTerminate
)

// String returns the stable label used in headers, metrics, and logs.
func (s Stage) String() string {
	switch s {
	case StageShapeTools:
		return "shape_tools"
	case StageCompress:
		return "compress"
	case StageDowngrade:
		return "downgrade"
	case StageTerminate:
		return "terminate"
	default:
		return "none"
	}
}

// Thresholds are ascending multipliers of the budget at which each stage fires.
// A session at ratio >= ShapeTools enters StageShapeTools, >= Compress enters
// StageCompress, and so on.
type Thresholds struct {
	ShapeTools float64
	Compress   float64
	Downgrade  float64
	Terminate  float64
}

// DefaultThresholds interprets the vision-paper "3x expected budget -> graduated
// response" as the terminate point, with softer stages below it.
func DefaultThresholds() Thresholds {
	return Thresholds{ShapeTools: 1.0, Compress: 1.5, Downgrade: 2.0, Terminate: 3.0}
}

// ResolveThresholds fills any zero field from DefaultThresholds, so config can
// override individual stages while leaving the rest at their defaults.
func ResolveThresholds(in Thresholds) Thresholds {
	def := DefaultThresholds()
	if in.ShapeTools == 0 {
		in.ShapeTools = def.ShapeTools
	}
	if in.Compress == 0 {
		in.Compress = def.Compress
	}
	if in.Downgrade == 0 {
		in.Downgrade = def.Downgrade
	}
	if in.Terminate == 0 {
		in.Terminate = def.Terminate
	}
	return in
}

// Evaluate returns the highest triggered stage and the over-budget ratio
// (cumulative/budget). A budget <= 0 or cumulative <= 0 disables enforcement and
// returns (StageNone, 0).
func Evaluate(cumulative, budget int64, t Thresholds) (Stage, float64) {
	if budget <= 0 || cumulative <= 0 {
		return StageNone, 0
	}
	ratio := float64(cumulative) / float64(budget)
	switch {
	case ratio >= t.Terminate:
		return StageTerminate, ratio
	case ratio >= t.Downgrade:
		return StageDowngrade, ratio
	case ratio >= t.Compress:
		return StageCompress, ratio
	case ratio >= t.ShapeTools:
		return StageShapeTools, ratio
	default:
		return StageNone, ratio
	}
}
