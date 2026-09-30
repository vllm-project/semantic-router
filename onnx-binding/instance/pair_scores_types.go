package instance

type PairScorerSelection struct {
	Layer     int `json:"layer"`
	Dimension int `json:"dimension"`
}

type TextPair struct {
	Query    string `json:"query"`
	Document string `json:"document"`
}

// PairScores preserves raw, uncalibrated logits and complete paired input usage.
type PairScores struct {
	Scores []float32    `json:"scores"`
	Inputs []InputUsage `json:"inputs"`
}
