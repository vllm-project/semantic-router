//go:build windows || !cgo || (!amd64 && !arm64 && !riscv64)

package candle_binding

func nativeInstanceLoadPairScorer(InstanceOptions, PairScorerSelection) (uint64, error) {
	return 0, ErrBackendUnavailable
}
func nativeInstanceScorePairs(uint64, []TextPair) (PairScores, error) {
	return PairScores{}, ErrBackendUnavailable
}
