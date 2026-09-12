package candle_binding

import (
	"errors"
	"fmt"
	"strings"
	"sync"
)

// InstanceOptions selects an owned model and the task's explicit input policy.
// ModelPath must be a prepared local artifact directory. Device defaults to cpu;
// accelerator selection is explicit and fails instead of falling back to CPU.
// Encoder and embedding loaders execute float32. Existing generative accelerator
// loaders use bfloat16. Encoder classification is limited to 512 tokens including
// special tokens; MaxInputTokens can reduce that budget.
type InstanceOptions struct {
	ModelPath           string            `json:"model_path"`
	ModelType           string            `json:"model_type,omitempty"`
	Device              string            `json:"device,omitempty"`
	Precision           string            `json:"precision,omitempty"`
	MaxInputTokens      int               `json:"max_input_tokens,omitempty"`
	Overflow            string            `json:"overflow,omitempty"`
	Adapters            []InstanceAdapter `json:"adapters,omitempty"`
	GenerationMaxTokens int               `json:"generation_max_tokens,omitempty"`
}

// InstanceAdapter is prepared during generative model loading. Adapter mutation
// after publication is deliberately not part of the owned instance API.
type InstanceAdapter struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

// InstanceInfo describes effective execution, not a requested device or a claim
// inferred from a checkpoint's name. ResourceID identifies the actual owned
// backbone. Explicit head bindings and clones retain the same ResourceID.
type InstanceInfo struct {
	ResourceID             uint64   `json:"resource_id"`
	Task                   string   `json:"task"`
	ModelType              string   `json:"model_type"`
	Device                 string   `json:"device"`
	Precision              string   `json:"precision"`
	ArchitecturalMaxTokens int      `json:"architectural_max_tokens"`
	MaxInputTokens         int      `json:"max_input_tokens"`
	Overflow               string   `json:"overflow"`
	Labels                 []string `json:"labels"`
	Modalities             []string `json:"modalities"`
}

// InputMetadata reports the complete input and what the task actually processed.
type InputMetadata struct {
	InputTokens     int  `json:"input_tokens"`
	ProcessedTokens int  `json:"processed_tokens"`
	Truncated       bool `json:"truncated"`
}

// DistributionOutput carries real softmax probabilities in Labels order.
type DistributionOutput struct {
	Class         int           `json:"class"`
	Confidence    float32       `json:"confidence"`
	Probabilities []float32     `json:"probabilities"`
	Labels        []string      `json:"labels"`
	Input         InputMetadata `json:"input"`
}

// InstanceSpan offsets are zero-based UTF-8 byte ranges in the caller's text.
type InstanceSpan struct {
	Text       string  `json:"text"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
	Confidence float32 `json:"confidence"`
	Label      string  `json:"label"`
}

type TokenOutput struct {
	Spans      []InstanceSpan `json:"spans"`
	Input      InputMetadata  `json:"input"`
	OffsetUnit string         `json:"offset_unit"`
}

// HallucinationOutput preserves the maintained detector's aggregate score;
// Confidence is not a calibrated probability. Span confidences are softmax
// scores, and their offsets refer to Answer rather than the formatted input.
type HallucinationOutput struct {
	HasHallucination bool           `json:"has_hallucination"`
	Confidence       float32        `json:"confidence"`
	Spans            []InstanceSpan `json:"spans"`
	Input            InputMetadata  `json:"input"`
	OffsetUnit       string         `json:"offset_unit"`
}

type InstanceEmbeddingOutput struct {
	Values []float32     `json:"values"`
	Input  InputMetadata `json:"input"`
}

var ErrInstanceClosed = errors.New("candle: native instance closed")

// InstanceError identifies an execution boundary without converting failures to
// model verdicts. Code is configuration, capability, input_limit, result_invalid,
// execution, or load. errors.Is(err, ErrInstanceClosed) detects closed handles.
type InstanceError struct {
	Code    string
	Message string
}

func (e *InstanceError) Error() string { return "candle " + e.Code + ": " + e.Message }

func instanceError(message string) error {
	code, detail, found := strings.Cut(message, ": ")
	if found {
		switch code {
		case "closed":
			return ErrInstanceClosed
		case "configuration", "capability", "input_limit", "result_invalid", "execution", "load":
			return &InstanceError{Code: code, Message: detail}
		}
	}
	return &InstanceError{Code: "load", Message: message}
}

// instance must not be copied. Holding RLock across CGo keeps Close synchronous
// with non-preemptible native work, even if the caller's context is cancelled.
type instance struct {
	mu     sync.RWMutex
	handle uint64
}

func (i *instance) Close() error {
	if i == nil {
		return nil
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.handle == 0 {
		return nil
	}
	err := nativeInstanceClose(i.handle)
	if err == nil {
		i.handle = 0
	}
	return err
}

func useInstance[T any](i *instance, call func(uint64) (T, error)) (T, error) {
	var zero T
	if i == nil {
		return zero, ErrInstanceClosed
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.handle == 0 {
		return zero, ErrInstanceClosed
	}
	return call(i.handle)
}

func (i *instance) Info() (InstanceInfo, error) { return useInstance(i, nativeInstanceInfo) }
func (i *instance) clone() (*instance, error) {
	h, err := useInstance(i, nativeInstanceClone)
	if err != nil {
		return nil, err
	}
	return &instance{handle: h}, nil
}

func loadInstance(options InstanceOptions, task string) (*instance, error) {
	if options.MaxInputTokens < 0 || options.GenerationMaxTokens < 0 {
		return nil, &InstanceError{Code: "configuration", Message: "negative input budget"}
	}
	h, err := nativeInstanceLoad(options, task)
	if err != nil {
		return nil, err
	}
	return &instance{handle: h}, nil
}

func (i *instance) bindHead(path, task string) (*instance, error) {
	h, err := useInstance(i, func(h uint64) (uint64, error) { return nativeInstanceBindHead(h, path, task) })
	if err != nil {
		return nil, err
	}
	return &instance{handle: h}, nil
}

type (
	SequenceClassifier    struct{ *instance }
	TokenClassifier       struct{ *instance }
	NLIClassifier         struct{ *instance }
	HallucinationDetector struct{ *instance }
	EmbeddingModel        struct{ *instance }
)

func LoadSequenceClassifier(options InstanceOptions) (*SequenceClassifier, error) {
	i, err := loadInstance(options, "sequence")
	if err != nil {
		return nil, err
	}
	return &SequenceClassifier{i}, nil
}

func LoadTokenClassifier(options InstanceOptions) (*TokenClassifier, error) {
	i, err := loadInstance(options, "token")
	if err != nil {
		return nil, err
	}
	return &TokenClassifier{i}, nil
}

func LoadNLIClassifier(options InstanceOptions) (*NLIClassifier, error) {
	i, err := loadInstance(options, "nli")
	if err != nil {
		return nil, err
	}
	return &NLIClassifier{i}, nil
}

func LoadHallucinationDetector(options InstanceOptions) (*HallucinationDetector, error) {
	i, err := loadInstance(options, "hallucination")
	if err != nil {
		return nil, err
	}
	return &HallucinationDetector{i}, nil
}

func LoadEmbeddingModel(options InstanceOptions) (*EmbeddingModel, error) {
	i, err := loadInstance(options, "embedding")
	if err != nil {
		return nil, err
	}
	return &EmbeddingModel{i}, nil
}

func (m *SequenceClassifier) Classify(text string) (DistributionOutput, error) {
	return useInstance(m.instance, func(h uint64) (DistributionOutput, error) { return nativeInstanceSequence(h, text) })
}

func (m *TokenClassifier) ClassifyTokens(text string) (TokenOutput, error) {
	return useInstance(m.instance, func(h uint64) (TokenOutput, error) { return nativeInstanceTokens(h, text) })
}

func (m *NLIClassifier) Classify(premise, hypothesis string) (DistributionOutput, error) {
	return useInstance(m.instance, func(h uint64) (DistributionOutput, error) { return nativeInstanceNLI(h, premise, hypothesis) })
}

func (m *HallucinationDetector) Detect(context, question, answer string, threshold float32) (HallucinationOutput, error) {
	return useInstance(m.instance, func(h uint64) (HallucinationOutput, error) {
		return nativeInstanceHallucination(h, context, question, answer, threshold)
	})
}

func (m *EmbeddingModel) Embed(text string, dimension int) (InstanceEmbeddingOutput, error) {
	return m.EmbedAtLayer(text, dimension, 0)
}

func (m *EmbeddingModel) EmbedAtLayer(text string, dimension, layer int) (InstanceEmbeddingOutput, error) {
	if dimension < 0 || layer < 0 {
		return InstanceEmbeddingOutput{}, fmt.Errorf("dimension and layer must be nonnegative")
	}
	return useInstance(m.instance, func(h uint64) (InstanceEmbeddingOutput, error) {
		return nativeInstanceEmbedding(h, text, dimension, layer)
	})
}

func (m *SequenceClassifier) Clone() (*SequenceClassifier, error) {
	i, err := m.instance.clone()
	if err != nil {
		return nil, err
	}
	return &SequenceClassifier{i}, nil
}

func (m *TokenClassifier) Clone() (*TokenClassifier, error) {
	i, err := m.instance.clone()
	if err != nil {
		return nil, err
	}
	return &TokenClassifier{i}, nil
}

func (m *NLIClassifier) Clone() (*NLIClassifier, error) {
	i, err := m.instance.clone()
	if err != nil {
		return nil, err
	}
	return &NLIClassifier{i}, nil
}

func (m *HallucinationDetector) Clone() (*HallucinationDetector, error) {
	i, err := m.instance.clone()
	if err != nil {
		return nil, err
	}
	return &HallucinationDetector{i}, nil
}

func (m *EmbeddingModel) Clone() (*EmbeddingModel, error) {
	i, err := m.instance.clone()
	if err != nil {
		return nil, err
	}
	return &EmbeddingModel{i}, nil
}

// BindSequenceHead loads only head weights and a private tokenizer/label mapping
// from path, while explicitly retaining this instance's backbone. Architecture
// compatibility is checked at load time; path equality never implies sharing.
func (m *SequenceClassifier) BindSequenceHead(path string) (*SequenceClassifier, error) {
	i, err := m.instance.bindHead(path, "sequence")
	if err != nil {
		return nil, err
	}
	return &SequenceClassifier{i}, nil
}

func (m *SequenceClassifier) BindTokenHead(path string) (*TokenClassifier, error) {
	i, err := m.instance.bindHead(path, "token")
	if err != nil {
		return nil, err
	}
	return &TokenClassifier{i}, nil
}

func (m *TokenClassifier) BindSequenceHead(path string) (*SequenceClassifier, error) {
	i, err := m.instance.bindHead(path, "sequence")
	if err != nil {
		return nil, err
	}
	return &SequenceClassifier{i}, nil
}

func (m *TokenClassifier) BindTokenHead(path string) (*TokenClassifier, error) {
	i, err := m.instance.bindHead(path, "token")
	if err != nil {
		return nil, err
	}
	return &TokenClassifier{i}, nil
}

// EmbedImage applies the maintained image decoder and SigLIP preprocessing to
// encoded PNG/JPEG bytes. Its weights share this handle's owned multimodal model.
func (m *EmbeddingModel) EmbedImage(encoded []byte, dimension int) (InstanceEmbeddingOutput, error) {
	if len(encoded) == 0 || dimension < 0 {
		return InstanceEmbeddingOutput{}, &InstanceError{Code: "configuration", Message: "invalid image or dimension"}
	}
	return useInstance(m.instance, func(h uint64) (InstanceEmbeddingOutput, error) { return nativeInstanceImage(h, encoded, dimension) })
}

// EmbedAudio encodes mel spectrogram values in row-major [melBins, frames] order.
func (m *EmbeddingModel) EmbedAudio(mel []float32, melBins, frames, dimension int) (InstanceEmbeddingOutput, error) {
	if melBins <= 0 || frames <= 0 || len(mel) == 0 || len(mel)/melBins != frames || len(mel)%melBins != 0 || dimension < 0 {
		return InstanceEmbeddingOutput{}, &InstanceError{Code: "configuration", Message: "invalid spectrogram shape or dimension"}
	}
	return useInstance(m.instance, func(h uint64) (InstanceEmbeddingOutput, error) {
		return nativeInstanceAudio(h, mel, melBins, frames, dimension)
	})
}

// Windows returns byte ranges covering every content token within this instance's
// input budget. Each range is suitable for embedding with the same tokenizer.
func (m *EmbeddingModel) Windows(text string, maxTokens int) ([]TextWindow, error) {
	if maxTokens < 0 {
		return nil, fmt.Errorf("maxTokens must be nonnegative")
	}
	ranges, err := useInstance(m.instance, func(handle uint64) ([][2]int, error) {
		return nativeInstanceTextWindows(handle, text, maxTokens)
	})
	if err != nil {
		return nil, err
	}
	windows := make([]TextWindow, len(ranges))
	for i, r := range ranges {
		windows[i] = TextWindow{Start: r[0], End: r[1]}
	}
	return windows, nil
}
