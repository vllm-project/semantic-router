package headers

// VSRBenchExpectedConfigHash pins an evaluation request to the SHA-256 of the
// active runtime document. It is consumed before any provider generation.
const VSRBenchExpectedConfigHash = "x-vsr-bench-expected-config-hash"

// VSRBenchMaxInferenceCalls limits the provider calls of an evaluation
// request. Its receipts ride in x-vsr-trace and contain no prompts,
// credentials, or content.
const VSRBenchMaxInferenceCalls = "x-vsr-bench-max-inference-calls"

// VSRConfigHash acknowledges the runtime generation a management request
// used: a routing preview or a configuration activation.
const VSRConfigHash = "x-vsr-config-hash"

// VSRConfigVersion names the version of the configuration snapshot that
// served.
const VSRConfigVersion = "x-vsr-config-version"

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
