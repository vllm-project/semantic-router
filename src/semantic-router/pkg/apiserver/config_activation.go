//go:build !windows && cgo

package apiserver

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"

type configActivationResponse struct {
	routerruntime.ConfigActivation
	Error string `json:"error,omitempty"`
}

func (s *ClassificationAPIServer) configActivation(hash string) *configActivationResponse {
	if s == nil || s.runtimeRegistry == nil {
		return nil
	}
	activation := s.runtimeRegistry.ConfigActivation()
	if activation.Attempt == 0 || activation.DocumentHash != hash {
		return nil
	}
	return &configActivationResponse{
		ConfigActivation: activation,
		Error:            scrubSecretsInErrorMessage(activation.FailureDetail),
	}
}

func (s *ClassificationAPIServer) configActivationStatus(hash, active string) string {
	if active != "" && active == hash {
		return "active"
	}
	if activation := s.configActivation(hash); activation != nil && activation.Status == "failed" {
		return "failed"
	}
	if active == "" {
		return "unknown"
	}
	return "pending"
}

func (s *ClassificationAPIServer) configActivationAttempt() uint64 {
	if s == nil || s.runtimeRegistry == nil {
		return 0
	}
	return s.runtimeRegistry.ConfigActivation().Attempt
}

// A persisted retry may have the same document hash as a failed attempt.
// Mutation results must describe preparation started after that retry, while
// GET /config/hash continues to describe the most recent observed attempt.
func (s *ClassificationAPIServer) configActivationAfter(hash string, afterAttempt uint64) *configActivationResponse {
	activation := s.configActivation(hash)
	if activation != nil && activation.Attempt <= afterAttempt {
		return nil
	}
	return activation
}
