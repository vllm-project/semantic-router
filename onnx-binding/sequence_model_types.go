package onnx_binding

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
