package candle_binding

// SequenceModelOptions is the exact contract of an independently owned head.
// Labels follow the artifact's id2label ordering. MultiLabel selects sigmoid;
// categorical heads use softmax. Inputs beyond MaxSequenceLength are rejected.
type SequenceModelOptions struct {
	ModelPath         string   `json:"model_path"`
	UseCPU            bool     `json:"use_cpu"`
	MaxSequenceLength int      `json:"max_sequence_length"`
	Labels            []string `json:"labels"`
	MultiLabel        bool     `json:"multi_label"`
}

// SequenceWindowOptions defines an exact token scan. Size includes the model's
// special tokens; Overlap counts content tokens shared by adjacent windows.
type SequenceWindowOptions struct {
	Size    int `json:"size"`
	Overlap int `json:"overlap"`
}

// SequenceWindowScores retains the full distribution for one content-token
// range [Start, End). Aggregate selected labels within each window before
// taking the maximum across windows; independent softmax maxima are invalid.
type SequenceWindowScores struct {
	Start  int       `json:"start"`
	End    int       `json:"end"`
	Scores []float32 `json:"scores"`
}
