// Package projectiontrace defines a versioned JSON payload for projection-layer
// explainability: partitions (exclusive / softmax winner selection), weighted scores,
// and threshold mappings. It is shared by classification, replay persistence, and
// dashboard/API consumers.
package projectiontrace

const SchemaVersion = "1"

// Trace captures per-request projection diagnostics for operators and replay.
type Trace struct {
	SchemaVersion string                `json:"schema_version"`
	Partitions    []PartitionResolution `json:"partitions,omitempty"`
	Scores        []ScoreBreakdown      `json:"scores,omitempty"`
	Mappings      []MappingDecision     `json:"mappings,omitempty"`
}

// PartitionContender is one candidate inside a projection partition before/after winner selection.
type PartitionContender struct {
	Name            string   `json:"name"`
	RawScore        float64  `json:"raw_score"`
	NormalizedScore *float64 `json:"normalized_score,omitempty"` // softmax weights; nil for non-softmax semantics
}

// PartitionResolution explains exclusive / softmax_exclusive partition reduction (winner + margins).
type PartitionResolution struct {
	GroupName      string               `json:"group_name"`
	SignalType     string               `json:"signal_type"`
	Semantics      string               `json:"semantics,omitempty"`
	Temperature    float64              `json:"temperature,omitempty"`
	Contenders     []PartitionContender `json:"contenders,omitempty"`
	Winner         string               `json:"winner,omitempty"`
	WinnerScore    float64              `json:"winner_score,omitempty"`     // value written to signal confidences after resolution
	RawWinnerScore float64              `json:"raw_winner_score,omitempty"` // confidence before softmax (same as raw for non-softmax)
	Margin         float64              `json:"margin,omitempty"`           // top − second among comparison scores (normalized if softmax)
	DefaultUsed    bool                 `json:"default_used,omitempty"`     // synthetic default member appended
}

// ScoreBreakdown is the weighted contribution decomposition for one projection score.
type ScoreBreakdown struct {
	Name   string           `json:"name"`
	Total  float64          `json:"total"`
	Inputs []ScoreInputPart `json:"inputs,omitempty"`
}

// ScoreInputPart is one weighted input to a projection score.
type ScoreInputPart struct {
	Type         string  `json:"type"`
	Name         string  `json:"name,omitempty"`
	KB           string  `json:"kb,omitempty"`
	Metric       string  `json:"metric,omitempty"`
	Weight       float64 `json:"weight"`
	Value        float64 `json:"value"`
	Contribution float64 `json:"contribution"`
}

// MappingDecision explains how one projection mapping turned a score into outputs.
type MappingDecision struct {
	MappingName      string           `json:"mapping_name"`
	SourceScore      string           `json:"source_score"`
	ScoreValue       float64          `json:"score_value"`
	SelectedOutput   string           `json:"selected_output,omitempty"`
	Confidence       float64          `json:"confidence,omitempty"`
	BoundaryDistance float64          `json:"boundary_distance,omitempty"`
	Outputs          []OutputEvalStep `json:"outputs,omitempty"`
}

// OutputEvalStep records threshold evaluation for one mapping output band.
type OutputEvalStep struct {
	Name             string  `json:"name"`
	Matched          bool    `json:"matched"`
	BoundaryDistance float64 `json:"boundary_distance"`
}
