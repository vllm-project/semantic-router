//go:build windows || !cgo || (!amd64 && !arm64)

package instance

type PairScorer struct{ *owner }

func LoadPairScorer(Options, PairScorerSelection) (*PairScorer, error) { return nil, unavailable }
func (*PairScorer) Clone() (*PairScorer, error)                        { return nil, unavailable }
func (*PairScorer) ScorePairs([]TextPair) (PairScores, error)          { return PairScores{}, unavailable }
