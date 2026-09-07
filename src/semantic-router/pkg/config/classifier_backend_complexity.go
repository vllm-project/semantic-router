package config

import "fmt"

// ComplexityBackendContracts are the response shapes the complexity signal can
// read. Both are supported because a difficulty model can be built either way,
// and the contract chosen decides how the runtime reads the response:
//
//   - score.v1 carries a continuous score, which the signal turns into a
//     verdict using each rule's declared boundaries.
//   - label_distribution.v1 carries the verdict directly as a label, so the
//     winning label is the verdict and its probability is the confidence.
//
// Neither is a substitute for the other, which is why the field cannot be
// defaulted for this signal.
var ComplexityBackendContracts = []string{
	RemoteClassifierContractScore,
	RemoteClassifierContractLabelDistribution,
}

// ValidateComplexityModelBackend checks the complexity signal's remote
// attachment. A nil backend is the local prototype-scoring path and stays
// valid, so this reports an error only for a configured backend the runtime
// could not honour.
func ValidateComplexityModelBackend(cfg *RouterConfig) error {
	if cfg == nil {
		return fmt.Errorf("complexity model configuration is nil")
	}
	backend := cfg.ComplexityModel.Backend
	if backend == nil {
		return nil
	}
	// http_chat returns prose. Reading either a score or a label distribution
	// out of it needs a parser this signal does not define, so it is rejected
	// here rather than failing per request.
	if backend.Protocol != RemoteClassifierProtocolHTTPClassify {
		return fmt.Errorf(
			"complexity.backend.protocol %q is not supported by the complexity consumer, use %q",
			backend.Protocol, RemoteClassifierProtocolHTTPClassify)
	}
	if _, err := ResolveRemoteClassifierBackend(
		cfg,
		backend,
		ModelRoleClassification,
		ComplexityBackendContracts...,
	); err != nil {
		return fmt.Errorf("complexity: %w", err)
	}
	return nil
}
