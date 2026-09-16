package candle_binding

// GuardOutput preserves the generated safety text. It carries no fabricated
// probability; callers interpret it using the maintained guard parser.
type GuardOutput struct {
	RawOutput string        `json:"raw_output"`
	Input     InputMetadata `json:"input"`
}

// GenerativeOutput records actual conditional label scores. The existing Qwen3
// adapter implementation currently applies templates and category mappings but
// does not apply LoRA deltas; AdapterWeightsApplied reports that fact explicitly.
type GenerativeOutput struct {
	DistributionOutput
	ScoreSemantics        string `json:"score_semantics"`
	AdapterWeightsApplied bool   `json:"adapter_weights_applied"`
}

type (
	GuardModel           struct{ *instance }
	GenerativeClassifier struct{ *instance }
)

// LoadGuardModel creates an owned Qwen3Guard, with mutable KV state serialized
// within this resource. Its budget includes the complete guard template and
// generation reserve. Overflow defaults to reject.
func LoadGuardModel(options InstanceOptions) (*GuardModel, error) {
	i, err := loadInstance(options, "guard")
	if err != nil {
		return nil, err
	}
	return &GuardModel{i}, nil
}

// LoadGenerativeClassifier prepares the existing Qwen3 label scorer and named
// adapters in a candidate instance. No process-global adapter slots are used.
func LoadGenerativeClassifier(options InstanceOptions) (*GenerativeClassifier, error) {
	i, err := loadInstance(options, "generative")
	if err != nil {
		return nil, err
	}
	return &GenerativeClassifier{i}, nil
}

func (m *GuardModel) Classify(text, mode string) (GuardOutput, error) {
	return useInstance(m.instance, func(h uint64) (GuardOutput, error) { return nativeInstanceGuard(h, text, mode) })
}

func (m *GuardModel) Clone() (*GuardModel, error) {
	i, err := m.instance.clone()
	if err != nil {
		return nil, err
	}
	return &GuardModel{i}, nil
}

func (m *GenerativeClassifier) Clone() (*GenerativeClassifier, error) {
	i, err := m.instance.clone()
	if err != nil {
		return nil, err
	}
	return &GenerativeClassifier{i}, nil
}

func (m *GenerativeClassifier) Classify(text, adapter string) (GenerativeOutput, error) {
	if adapter == "" {
		return GenerativeOutput{}, &InstanceError{Code: "configuration", Message: "adapter name is required"}
	}
	return useInstance(m.instance, func(h uint64) (GenerativeOutput, error) {
		return nativeInstanceGenerative(h, text, adapter, nil, false)
	})
}

func (m *GenerativeClassifier) ClassifyZeroShot(text string, categories []string) (GenerativeOutput, error) {
	return useInstance(m.instance, func(h uint64) (GenerativeOutput, error) {
		return nativeInstanceGenerative(h, text, "", categories, false)
	})
}

func (m *GenerativeClassifier) ClassifyZeroShotMultiToken(text string, categories []string) (GenerativeOutput, error) {
	return useInstance(m.instance, func(h uint64) (GenerativeOutput, error) {
		return nativeInstanceGenerative(h, text, "", categories, true)
	})
}
