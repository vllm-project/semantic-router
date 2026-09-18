package headers

// SRBenchExpectedConfigHash pins an evaluation request to the SHA-256 of the
// active runtime document. It is consumed before any provider generation.
const SRBenchExpectedConfigHash = "x-sr-bench-expected-config-hash"

// VSRConfigHash acknowledges the actual request-owned runtime generation.
const VSRConfigHash = "x-vsr-config-hash"

func ValidConfigHash(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, char := range value {
		switch {
		case char >= '0' && char <= '9', char >= 'a' && char <= 'f':
		default:
			return false
		}
	}
	return true
}

// Benchmark accounting receipts contain no prompts, credentials, or content.
const (
	SRBenchMaxInferenceCalls = "x-sr-bench-max-inference-calls"
	VSRInferenceCallCount    = "x-vsr-inference-call-count"
	VSRModelUsage            = "x-vsr-model-usage"
)
