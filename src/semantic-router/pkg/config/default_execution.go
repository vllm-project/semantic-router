package config

import "runtime"

// DefaultModelExecution is the compatibility policy for catalog entries without an
// explicit deployment. Packaging selects defaultModelProvider; the same binary still
// supports explicit Candle and ORT bindings. Requests never select a provider.
func DefaultModelExecution(useCPU bool) (provider, device string) {
	provider, device = defaultModelProvider, "cpu"
	if useCPU {
		return provider, device
	}
	if provider == "ort" {
		return provider, "migraphx:0"
	}
	if runtime.GOOS == "darwin" {
		return provider, "metal:0"
	}
	return provider, "cuda:0"
}
