// Package instance exposes independently owned, typed ONNX Runtime tasks.
// It can be linked alongside Candle when ORT is built without legacy-ffi.
package instance

// Options fixes execution and input policy before a model is loaded.
type Options struct {
	ModelPath               string `json:"model_path"`
	ModelFile               string `json:"model_file,omitempty"`
	Provider                string `json:"provider,omitempty"` // cpu (default), migraphx, or rocm
	DeviceID                int    `json:"device_id,omitempty"`
	Precision               string `json:"precision,omitempty"` // native (default) or MIGraphX fp16 conversion
	MaxInputTokens          int    `json:"max_input_tokens,omitempty"`
	DocumentMaxInputTokens  int    `json:"document_max_input_tokens,omitempty"` // typed window tasks only
	ExecutionMaxInputTokens int    `json:"execution_max_input_tokens,omitempty"`
	CompilationCacheDir     string `json:"compilation_cache_dir,omitempty"`
	Overflow                string `json:"overflow,omitempty"` // reject (default) or truncate_right
	IntraThreads            int    `json:"intra_threads,omitempty"`
	ProfilePrefix           string `json:"profile_prefix,omitempty"`
	CustomOpsProfile        string `json:"custom_ops_profile,omitempty"` // none (default) or ck_flash_attention
	AllowCPUFallback        bool   `json:"allow_cpu_fallback,omitempty"` // ROCm only; requires an audited ORT profile
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
// InputSchema records actual loaded session declarations for every provider;
// dynamic dimensions are -1 and fixed dimensions remain explicit.
type SessionEvidence struct {
	RuntimeBuild            string                    `json:"runtime_build"`
	CompilerFlags           map[string]string         `json:"compiler_flags"`
	Artifacts               []ArtifactDigest          `json:"artifacts"`
	ExecutionMaxInputTokens int                       `json:"execution_max_input_tokens,omitempty"`
	ExecutionInputs         []ExecutionInput          `json:"execution_inputs"`
	InputSchema             []ExecutionInput          `json:"input_schema"`
	CompilationCache        *CompilationCacheEvidence `json:"compilation_cache,omitempty"`
	Graph                   string                    `json:"graph"`
	Provider                string                    `json:"provider"`
	DeviceID                int                       `json:"device_id"`
	Precision               string                    `json:"precision"`
	CPUFallbackDisabled     bool                      `json:"cpu_fallback_disabled"`
	CustomOpsProfile        string                    `json:"custom_ops_profile"`
	CustomOpsLibrary        string                    `json:"custom_ops_library,omitempty"`
	CustomOpsSHA256         string                    `json:"custom_ops_sha256,omitempty"`
	ProfilePrefix           string                    `json:"profile_prefix"`
}

// ArtifactDigest is captured from files actually consumed by the owned session.
type ArtifactDigest struct {
	Role   string `json:"role"`
	SHA256 string `json:"sha256"`
}

// ExecutionInput describes one named tensor input. In SessionEvidence.InputSchema,
// its shape preserves the loaded graph's declared dimensions, including -1 for
// dynamic axes. ExecutionInputs separately records an explicitly fixed contract;
// an empty ExecutionInputs slice does not imply a dynamic graph.
type ExecutionInput struct {
	Name  string  `json:"name"`
	Dtype string  `json:"dtype"`
	Shape []int64 `json:"shape"`
}

type CompilationCacheEvidence struct {
	Key               string            `json:"key"`
	State             string            `json:"state"`
	Files             map[string]string `json:"files"`
	CompiledFileReads []string          `json:"compiled_file_reads"`
}

type AudioCapability struct {
	SampleRates   []int  `json:"sample_rates"`
	MaxSampleRate int    `json:"max_sample_rate"`
	MaxSeconds    int    `json:"max_seconds"`
	MaxChannels   int    `json:"max_channels"`
	Layout        string `json:"layout"`
}

type Info struct {
	Modalities             []string             `json:"modalities,omitempty"`
	Pooling                string               `json:"pooling,omitempty"`
	Normalization          string               `json:"normalization,omitempty"`
	AvailableDimensions    []int                `json:"available_dimensions,omitempty"`
	Audio                  *AudioCapability     `json:"audio,omitempty"`
	PairScorer             *PairScorerSelection `json:"pair_scorer,omitempty"`
	Task                   string               `json:"task"`
	ModelLimit             int                  `json:"model_limit"`
	TaskLimit              int                  `json:"task_limit"`
	EffectiveLimit         int                  `json:"effective_limit"`
	DocumentMaxInputTokens int                  `json:"document_max_input_tokens"`
	Overflow               string               `json:"overflow"`
	Labels                 []string             `json:"labels"`
	Dimension              int                  `json:"dimension"`
	AvailableLayers        []int                `json:"available_layers"`
	Sessions               []SessionEvidence    `json:"sessions"`
	CompletedInferences    uint64               `json:"completed_inferences"`
}

// TextWindow is a UTF-8 byte range in the original input (End exclusive).
type TextWindow struct {
	Start int `json:"start"`
	End   int `json:"end"`
}
