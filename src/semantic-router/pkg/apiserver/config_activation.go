//go:build !windows

package apiserver

import (
	"net/http"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

type configActivationResponse struct {
	routerruntime.ConfigActivation
	Error string `json:"error,omitempty"`
}

// redactedActivation is an activation as the management boundary serves it:
// diagnostics are scrubbed of credential values.
func redactedActivation(activation routerruntime.ConfigActivation) *configActivationResponse {
	reasons := make([]configsnapshot.Reason, len(activation.Reasons))
	for i, reason := range activation.Reasons {
		reason.Message = scrubSecretsInErrorMessage(reason.Message)
		reasons[i] = reason
	}
	if len(reasons) == 0 {
		reasons = nil
	}
	activation.Reasons = reasons
	return &configActivationResponse{
		ConfigActivation: activation,
		Error:            scrubSecretsInErrorMessage(activation.FailureDetail),
	}
}

func (s *ClassificationAPIServer) configActivation(hash string) *configActivationResponse {
	if s == nil || s.runtimeRegistry == nil {
		return nil
	}
	activation := s.runtimeRegistry.ConfigActivation()
	if activation.Attempt == 0 || activation.DocumentHash != hash {
		return nil
	}
	return redactedActivation(activation)
}

// lastConfigRejection is the most recent rejected update, which stays
// visible after later updates succeed.
func (s *ClassificationAPIServer) lastConfigRejection() *configActivationResponse {
	if s == nil || s.runtimeRegistry == nil {
		return nil
	}
	rejection, ok := s.runtimeRegistry.LastConfigRejection()
	if !ok {
		return nil
	}
	return redactedActivation(rejection)
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

// activeConfigSnapshot is the configuration snapshot that serves, or nil
// before the runtime published one.
func (s *ClassificationAPIServer) activeConfigSnapshot() *configsnapshot.Snapshot {
	if s == nil || s.runtimeRegistry == nil {
		return nil
	}
	return s.runtimeRegistry.ConfigSnapshot()
}

// activeConfigVersion is the version that serves the document with hash, or 0
// when another document serves.
func (s *ClassificationAPIServer) activeConfigVersion(hash string) uint64 {
	snapshot := s.activeConfigSnapshot()
	if snapshot == nil || hash == "" || snapshot.Hash() != hash {
		return 0
	}
	return snapshot.Version()
}

// setActiveConfigHeaders names the snapshot that serves. It can differ from a
// persisted document that is still activating or was rejected.
func (s *ClassificationAPIServer) setActiveConfigHeaders(w http.ResponseWriter) {
	snapshot := s.activeConfigSnapshot()
	if snapshot == nil {
		return
	}
	w.Header().Set(headers.VSRConfigVersion, strconv.FormatUint(snapshot.Version(), 10))
	if snapshot.Hash() != "" {
		w.Header().Set(headers.VSRConfigHash, snapshot.Hash())
	}
}
