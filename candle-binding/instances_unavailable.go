//go:build windows || !cgo || (!amd64 && !arm64)

package candle_binding

func nativeInstanceLoad(InstanceOptions, string) (uint64, error)    { return 0, ErrBackendUnavailable }
func nativeInstanceClose(uint64) error                              { return ErrBackendUnavailable }
func nativeInstanceClone(uint64) (uint64, error)                    { return 0, ErrBackendUnavailable }
func nativeInstanceInfo(uint64) (InstanceInfo, error)               { return InstanceInfo{}, ErrBackendUnavailable }
func nativeInstanceBindHead(uint64, string, string) (uint64, error) { return 0, ErrBackendUnavailable }
func nativeInstanceSequence(uint64, string) (DistributionOutput, error) {
	return DistributionOutput{}, ErrBackendUnavailable
}

func nativeInstanceTokens(uint64, string) (TokenOutput, error) {
	return TokenOutput{}, ErrBackendUnavailable
}

func nativeInstanceNLI(uint64, string, string) (DistributionOutput, error) {
	return DistributionOutput{}, ErrBackendUnavailable
}

func nativeInstanceHallucination(uint64, string, string, string, float32) (HallucinationOutput, error) {
	return HallucinationOutput{}, ErrBackendUnavailable
}

func nativeInstanceEmbedding(uint64, string, int, int) (InstanceEmbeddingOutput, error) {
	return InstanceEmbeddingOutput{}, ErrBackendUnavailable
}

func nativeInstanceImage(uint64, []byte, int) (InstanceEmbeddingOutput, error) {
	return InstanceEmbeddingOutput{}, ErrBackendUnavailable
}

func nativeInstanceAudio(uint64, []float32, int, int, int) (InstanceEmbeddingOutput, error) {
	return InstanceEmbeddingOutput{}, ErrBackendUnavailable
}

func nativeInstanceGuard(uint64, string, string) (GuardOutput, error) {
	return GuardOutput{}, ErrBackendUnavailable
}

func nativeInstanceGenerative(uint64, string, string, []string, bool) (GenerativeOutput, error) {
	return GenerativeOutput{}, ErrBackendUnavailable
}

func nativeInstanceTextWindows(uint64, string, int) ([][2]int, error) {
	return nil, ErrBackendUnavailable
}
