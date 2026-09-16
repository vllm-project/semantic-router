package candle_binding

import "unicode/utf8"

// PairScorerSelection fixes one trained encoder exit. Zero requests the actual
// full layer or width; Info reports the resolved selection after loading.
type PairScorerSelection struct {
	Layer     int `json:"layer"`
	Dimension int `json:"dimension"`
}

type TextPair struct {
	Query    string `json:"query"`
	Document string `json:"document"`
}

// PairScores contains raw, uncalibrated relevance logits in input order.
type PairScores struct {
	Scores []float32       `json:"scores"`
	Inputs []InputMetadata `json:"inputs"`
}

type PairScorer struct{ *instance }

func LoadPairScorer(options InstanceOptions, selection PairScorerSelection) (*PairScorer, error) {
	if options.MaxInputTokens < 0 || options.GenerationMaxTokens < 0 || selection.Layer < 0 || selection.Dimension < 0 {
		return nil, &InstanceError{Code: "configuration", Message: "negative pair scorer budget or selection"}
	}
	handle, err := nativeInstanceLoadPairScorer(options, selection)
	if err != nil {
		return nil, err
	}
	return &PairScorer{&instance{handle: handle}}, nil
}

func (m *PairScorer) Clone() (*PairScorer, error) {
	if m == nil {
		return nil, ErrInstanceClosed
	}
	i, err := m.instance.clone()
	if err != nil {
		return nil, err
	}
	return &PairScorer{i}, nil
}

func (m *PairScorer) ScorePairs(pairs []TextPair) (PairScores, error) {
	if m == nil {
		return PairScores{}, ErrInstanceClosed
	}
	for _, pair := range pairs {
		if !utf8.ValidString(pair.Query) || !utf8.ValidString(pair.Document) {
			return PairScores{}, &InstanceError{Code: "configuration", Message: "pair input must be valid UTF-8"}
		}
	}
	return useInstance(m.instance, func(h uint64) (PairScores, error) { return nativeInstanceScorePairs(h, pairs) })
}
