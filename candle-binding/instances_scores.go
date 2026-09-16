package candle_binding

// SequenceWindowOptions includes special tokens in Size; Overlap counts content
// tokens shared by adjacent windows. The complete input budget is independent.
type SequenceWindowOptions struct {
	Size    int `json:"size"`
	Overlap int `json:"overlap"`
}

// LabelScorer owns an independent-label task. Score uses the checkpoint's
// sigmoid contract; it cannot be called through categorical Classify.
type LabelScorer struct{ *instance }

type LabelScoresOutput struct {
	Scores []float32     `json:"scores"`
	Labels []string      `json:"labels"`
	Input  InputMetadata `json:"input"`
}

// Window coordinates are half-open offsets in the original content tokens,
// excluding the fixed special-token prefix and suffix.
type ClassificationWindow struct {
	Start         int       `json:"start"`
	End           int       `json:"end"`
	Probabilities []float32 `json:"probabilities"`
}

type LabelScoreWindow struct {
	Start  int       `json:"start"`
	End    int       `json:"end"`
	Scores []float32 `json:"scores"`
}

type WindowedClassificationOutput struct {
	Windows       []ClassificationWindow `json:"windows"`
	Labels        []string               `json:"labels"`
	ContentTokens int                    `json:"content_tokens"`
	Input         InputMetadata          `json:"input"`
}

type WindowedLabelScoresOutput struct {
	Windows       []LabelScoreWindow `json:"windows"`
	Labels        []string           `json:"labels"`
	ContentTokens int                `json:"content_tokens"`
	Input         InputMetadata      `json:"input"`
}

func LoadLabelScorer(options InstanceOptions) (*LabelScorer, error) {
	i, err := loadInstance(options, "label_scores")
	if err != nil {
		return nil, err
	}
	return &LabelScorer{i}, nil
}

func (m *LabelScorer) Clone() (*LabelScorer, error) {
	i, err := m.instance.clone()
	if err != nil {
		return nil, err
	}
	return &LabelScorer{i}, nil
}

func (m *LabelScorer) Score(text string) (LabelScoresOutput, error) {
	return useInstance(m.instance, func(h uint64) (LabelScoresOutput, error) { return nativeInstanceScore(h, text) })
}

func validSequenceWindow(options SequenceWindowOptions) error {
	if options.Size <= 0 || options.Overlap < 0 || options.Overlap >= options.Size {
		return &InstanceError{Code: "configuration", Message: "window requires positive size and smaller nonnegative overlap"}
	}
	return nil
}

func (m *SequenceClassifier) ClassifyWindows(text string, options SequenceWindowOptions) (WindowedClassificationOutput, error) {
	if err := validSequenceWindow(options); err != nil {
		return WindowedClassificationOutput{}, err
	}
	return useInstance(m.instance, func(h uint64) (WindowedClassificationOutput, error) {
		return nativeInstanceClassifyWindows(h, text, options)
	})
}

func (m *LabelScorer) ScoreWindows(text string, options SequenceWindowOptions) (WindowedLabelScoresOutput, error) {
	if err := validSequenceWindow(options); err != nil {
		return WindowedLabelScoresOutput{}, err
	}
	return useInstance(m.instance, func(h uint64) (WindowedLabelScoresOutput, error) { return nativeInstanceScoreWindows(h, text, options) })
}

func (m *Backbone) BindLabelScoreHead(path string) (*LabelScorer, error) {
	i, err := m.instance.bindHead(path, "label_scores")
	if err != nil {
		return nil, err
	}
	return &LabelScorer{i}, nil
}

func (m *SequenceClassifier) BindLabelScoreHead(path string) (*LabelScorer, error) {
	i, err := m.instance.bindHead(path, "label_scores")
	if err != nil {
		return nil, err
	}
	return &LabelScorer{i}, nil
}

func (m *TokenClassifier) BindLabelScoreHead(path string) (*LabelScorer, error) {
	i, err := m.instance.bindHead(path, "label_scores")
	if err != nil {
		return nil, err
	}
	return &LabelScorer{i}, nil
}

// WindowedTokenOutput contains globally decoded spans and exact coverage.
type WindowedTokenOutput struct {
	TokenOutput
	ContentTokens int      `json:"content_tokens"`
	Windows       [][2]int `json:"windows"`
}

func (m *TokenClassifier) ClassifyWindows(text string, options SequenceWindowOptions) (WindowedTokenOutput, error) {
	if err := validSequenceWindow(options); err != nil {
		return WindowedTokenOutput{}, err
	}
	return useInstance(m.instance, func(h uint64) (WindowedTokenOutput, error) { return nativeInstanceTokenWindows(h, text, options) })
}
