// Package instance exposes independently owned, typed ONNX Runtime tasks.
// It can be linked alongside Candle when ORT is built without legacy-ffi.
package instance

// Options fixes execution and input policy before a model is loaded.
type Options struct {
	ModelPath      string `json:"model_path"`
	ModelFile      string `json:"model_file,omitempty"`
	Provider       string `json:"provider,omitempty"` // cpu (default) or migraphx
	DeviceID       int    `json:"device_id,omitempty"`
	Precision      string `json:"precision,omitempty"` // native (default) or MIGraphX fp16 conversion
	MaxInputTokens int    `json:"max_input_tokens,omitempty"`
	Overflow       string `json:"overflow,omitempty"` // reject (default) or truncate_right
	IntraThreads   int    `json:"intra_threads,omitempty"`
	ProfilePrefix  string `json:"profile_prefix,omitempty"`
}

type InstanceOptions = Options

// Error reports the provider boundary that failed. Kind is stable; Message is
// diagnostic text and may contain local artifact paths, so must be redacted for public logs.
type Error struct {
	Kind    string
	Message string
}

func (e *Error) Error() string { return "ort " + e.Kind + ": " + e.Message }

type InputUsage struct {
	OriginalTokens  int  `json:"original_tokens"`
	ProcessedTokens int  `json:"processed_tokens"`
	Truncated       bool `json:"truncated"`
}

// Distribution contains the complete ordered label probabilities. Confidence is
// the selected label's probability, without any claim of external calibration.
type Distribution struct {
	Label         string     `json:"label"`
	ClassID       int        `json:"class_id"`
	Confidence    float32    `json:"confidence"`
	Labels        []string   `json:"labels"`
	Probabilities []float32  `json:"probabilities"`
	Input         InputUsage `json:"input"`
}

type Span struct {
	Text       string  `json:"text"`
	EntityType string  `json:"entity_type"`
	Start      int     `json:"start"` // UTF-8 byte offset, inclusive
	End        int     `json:"end"`   // UTF-8 byte offset, exclusive
	Confidence float32 `json:"confidence"`
}

type TokenSpans struct {
	Spans      []Span     `json:"spans"`
	OffsetUnit string     `json:"offset_unit"`
	Input      InputUsage `json:"input"`
}

type EmbeddingResult struct {
	Values          []float32   `json:"values"`
	Normalized      bool        `json:"normalized"`
	Modality        string      `json:"modality"`
	Input           *InputUsage `json:"input"`
	TruncatedFrames bool        `json:"truncated_frames"`
}

// SessionEvidence describes loaded execution policy. Successful registration is
// not an inference claim; CompletedInferences and the ORT profile prove execution.
type SessionEvidence struct {
	RuntimeBuild        string `json:"runtime_build"`
	Graph               string `json:"graph"`
	Provider            string `json:"provider"`
	DeviceID            int    `json:"device_id"`
	Precision           string `json:"precision"`
	CPUFallbackDisabled bool   `json:"cpu_fallback_disabled"`
	ProfilePrefix       string `json:"profile_prefix"`
}

type Info struct {
	Task                string            `json:"task"`
	ModelLimit          int               `json:"model_limit"`
	TaskLimit           int               `json:"task_limit"`
	EffectiveLimit      int               `json:"effective_limit"`
	Overflow            string            `json:"overflow"`
	Labels              []string          `json:"labels"`
	Dimension           int               `json:"dimension"`
	Sessions            []SessionEvidence `json:"sessions"`
	CompletedInferences uint64            `json:"completed_inferences"`
}

// TextWindow is a UTF-8 byte range in the original input (End exclusive).
type TextWindow struct {
	Start int `json:"start"`
	End   int `json:"end"`
}
