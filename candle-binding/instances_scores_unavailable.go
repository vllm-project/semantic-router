//go:build windows || !cgo || (!amd64 && !arm64 && !riscv64)

package candle_binding

func nativeInstanceScore(uint64, string) (LabelScoresOutput, error) {
	return LabelScoresOutput{}, ErrBackendUnavailable
}
func nativeInstanceClassifyWindows(uint64, string, SequenceWindowOptions) (WindowedClassificationOutput, error) {
	return WindowedClassificationOutput{}, ErrBackendUnavailable
}
func nativeInstanceScoreWindows(uint64, string, SequenceWindowOptions) (WindowedLabelScoresOutput, error) {
	return WindowedLabelScoresOutput{}, ErrBackendUnavailable
}

func nativeInstanceTokenWindows(uint64, string, SequenceWindowOptions) (WindowedTokenOutput, error) {
	return WindowedTokenOutput{}, ErrBackendUnavailable
}
