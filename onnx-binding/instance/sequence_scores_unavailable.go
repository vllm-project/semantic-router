//go:build windows || !cgo || (!amd64 && !arm64)

package instance

type LabelScorer struct{ *owner }

func LoadLabelScorer(Options) (*LabelScorer, error)    { return nil, unavailable }
func (*LabelScorer) Clone() (*LabelScorer, error)      { return nil, unavailable }
func (*LabelScorer) Score(string) (LabelScores, error) { return LabelScores{}, unavailable }
func (*LabelScorer) ScoreWindows(string, SequenceWindowOptions) (WindowedLabelScoresOutput, error) {
	return WindowedLabelScoresOutput{}, unavailable
}
func (*SequenceClassifier) ClassifyWindows(string, SequenceWindowOptions) (WindowedClassificationOutput, error) {
	return WindowedClassificationOutput{}, unavailable
}

func (*TokenClassifier) DetectWindows(string, SequenceWindowOptions) (WindowedTokenOutput, error) {
	return WindowedTokenOutput{}, unavailable
}
