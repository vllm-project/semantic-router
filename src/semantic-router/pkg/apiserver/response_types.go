//go:build !windows && cgo

package apiserver

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"

type healthResponse struct {
	Status  string `json:"status"`
	Service string `json:"service"`
}

type readinessResponse struct {
	Status  string `json:"status"`
	Service string `json:"service"`
	Ready   bool   `json:"ready"`
	*readinessDetails
}

type readinessDetails struct {
	Phase            string   `json:"phase"`
	Message          string   `json:"message"`
	DownloadingModel string   `json:"downloading_model"`
	PendingModels    []string `json:"pending_models"`
	ReadyModels      int      `json:"ready_models"`
	TotalModels      int      `json:"total_models"`
}

// objectListResponse is the existing OpenAI-compatible collection envelope.
type objectListResponse[T any] struct {
	Object string `json:"object"`
	Data   []T    `json:"data"`
}

type objectDeletedResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

type recipeValidationResponse struct {
	Valid  bool   `json:"valid"`
	Name   string `json:"name"`
	Action string `json:"action"`
}

type cacheHealthResponse struct {
	Status       string                    `json:"status"`
	Capabilities cache.BackendCapabilities `json:"capabilities"`
}

type compressionHealthResponse struct {
	Status   string `json:"status"`
	Recovery string `json:"recovery"`
}

type recoveryInvalidationResponse struct {
	Deleted int64 `json:"deleted"`
}

// managementErrorResponse is shared by every management error writer.
// Timestamp is optional for optimistic-concurrency failures created before
// a server instance is available; request_id is available through middleware.
type managementErrorResponse struct {
	Error managementErrorDetail `json:"error"`
}
type managementErrorDetail struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Timestamp string `json:"timestamp,omitempty"`
	RequestID string `json:"request_id,omitempty"`
}
