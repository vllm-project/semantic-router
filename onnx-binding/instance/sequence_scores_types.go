package instance

// SequenceWindowOptions counts special tokens in Size and content tokens in
// Overlap. Windows cover the admitted original token sequence without decoding.
type SequenceWindowOptions struct {
	Size    int
	Overlap int
}

type LabelScores struct {
	Scores []float32  `json:"scores"`
	Labels []string   `json:"labels"`
	Input  InputUsage `json:"input"`
}

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
	Input         InputUsage             `json:"input"`
}

type WindowedLabelScoresOutput struct {
	Windows       []LabelScoreWindow `json:"windows"`
	Labels        []string           `json:"labels"`
	ContentTokens int                `json:"content_tokens"`
	Input         InputUsage         `json:"input"`
}

func validSequenceWindow(options SequenceWindowOptions) error {
	if options.Size <= 0 || options.Overlap < 0 || options.Overlap >= options.Size {
		return &Error{Kind: "configuration", Message: "window requires positive size and smaller nonnegative overlap"}
	}
	return nil
}

type WindowedTokenOutput struct {
	TokenSpans
	ContentTokens int      `json:"content_tokens"`
	Windows       [][2]int `json:"windows"`
}
