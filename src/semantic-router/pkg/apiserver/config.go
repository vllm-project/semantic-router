//go:build !windows

package apiserver

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelinventory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicmodels"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

// ClassificationAPIServer holds the server state and dependencies
type ClassificationAPIServer struct {
	classificationSvc    classificationService
	previewAdmissionOnce sync.Once
	previewAdmission     admission.Admissioner
	configMu             sync.RWMutex
	config               *config.RouterConfig
	runtimeConfig        *liveRuntimeConfig
	runtimeRegistry      *routerruntime.Registry
	configPath           string // path to the router config file (for read/update/rollback)
	gatewayMode          config.GatewayMode
	memoryStore          memory.Store
	startupStateLoader   func() *startupstatus.State
	// The startup-status writer is created once during process bootstrap. Keep
	// its storage contract stable across live config swaps so /ready does not
	// start reading from a different backend after a successful reload.
	startupStatusConfig     *config.StartupStatusConfig
	responseCache           *cache.ResponseCacheService
	contextCompression      *contextcompression.Service
	compressionRecovery     contextcompression.RecoveryStore
	managementAuditMu       sync.Mutex
	managementAuditEntries  []managementAuditEntry
	managementAuditLastHash string
	managementAuditSequence uint64
	// learningOutcomePolicy gates POST /api/v1/observability/outcomes (idempotency + rate limit).
	learningOutcomePolicyOnce sync.Once
	learningOutcomePolicy     *learningOutcomeIngestPolicy
}

func (s *ClassificationAPIServer) currentContextCompression() (
	*contextcompression.Service,
	contextcompression.RecoveryStore,
	func(),
) {
	if s == nil {
		return nil, nil, func() {}
	}
	if s.runtimeRegistry != nil {
		service, recovery, release := s.runtimeRegistry.AcquireContextCompression()
		if service != nil {
			return service, recovery, release
		}
		release()
		return nil, nil, func() {}
	}
	return s.contextCompression, s.compressionRecovery, func() {}
}

func (s *ClassificationAPIServer) currentResponseCache() (*cache.ResponseCacheService, func()) {
	if s == nil {
		return nil, func() {}
	}
	if s.runtimeRegistry != nil {
		if service, release := s.runtimeRegistry.AcquireResponseCache(); service != nil {
			return service, release
		} else {
			release()
		}
		return nil, func() {}
	}
	return s.responseCache, func() {}
}

type (
	ModelsInfoResponse = modelinventory.ModelsInfoResponse
	ModelsInfoSummary  = modelinventory.ModelsInfoSummary
	ModelInfo          = modelinventory.ModelInfo
	ModelRegistryInfo  = config.ModelRegistryInfo
	SystemInfo         = modelinventory.SystemInfo
	OpenAIModel        = publicmodels.OpenAIModel
	OpenAIModelList    = publicmodels.OpenAIModelList
)

// BatchClassificationRequest represents a batch classification request
type BatchClassificationRequest struct {
	Recipe   string                         `json:"recipe,omitempty"`
	Texts    []string                       `json:"texts,omitempty"`
	Inputs   []services.ClassificationInput `json:"inputs,omitempty"`
	TaskType string                         `json:"task_type,omitempty"` // "intent" (default), "pii", "security", or "all" (intent fields plus pii and security objects)
	Options  *ClassificationOptions         `json:"options,omitempty"`
}

// BatchClassificationResult represents a single classification result with optional probabilities
type BatchClassificationResult struct {
	Index               *int                              `json:"index,omitempty"`
	Error               *services.ClassificationItemError `json:"error,omitempty"`
	MatchedSignals      *services.MatchedSignals          `json:"matched_signals,omitempty"`
	ConfidenceAvailable *bool                             `json:"confidence_available,omitempty"`
	Category            string                            `json:"category"`
	Confidence          float64                           `json:"confidence"`
	ProcessingTimeMs    int64                             `json:"processing_time_ms"`
	Probabilities       map[string]float64                `json:"probabilities,omitempty"`
	// PII and Security are only set for task_type "all".
	PII      *BatchPIIResult      `json:"pii,omitempty"`
	Security *BatchSecurityResult `json:"security,omitempty"`
}

// BatchPIIResult is the PII detection outcome for one text in an "all" batch.
// Confidence is null when the model did not report scores, mirroring the
// encoding of classification.PIIResult.
type BatchPIIResult struct {
	HasPII          bool     `json:"has_pii"`
	PIITypes        []string `json:"pii_types,omitempty"`
	Confidence      *float32 `json:"confidence"`
	ScoresAvailable *bool    `json:"scores_available,omitempty"`
}

// BatchSecurityResult is the jailbreak detection outcome for one text in an
// "all" batch. Confidence is null when the model did not report scores,
// mirroring the encoding of classification.SecurityResult.
type BatchSecurityResult struct {
	IsJailbreak     bool     `json:"is_jailbreak"`
	ThreatType      string   `json:"threat_type"`
	Confidence      *float32 `json:"confidence"`
	ScoresAvailable *bool    `json:"scores_available,omitempty"`
}

// BatchClassificationResponse represents the response from batch classification
type BatchClassificationResponse struct {
	Recipe           string                           `json:"recipe,omitempty"`
	Results          []BatchClassificationResult      `json:"results"`
	TotalCount       int                              `json:"total_count"`
	ProcessingTimeMs int64                            `json:"processing_time_ms"`
	Statistics       CategoryClassificationStatistics `json:"statistics"`
}

// CategoryClassificationStatistics provides batch processing statistics
type CategoryClassificationStatistics struct {
	CategoryDistribution map[string]int `json:"category_distribution"`
	AvgConfidence        float64        `json:"avg_confidence"`
	LowConfidenceCount   int            `json:"low_confidence_count"`
}

// ClassificationOptions mirrors services.IntentOptions for API layer
type ClassificationOptions struct {
	ReturnProbabilities bool    `json:"return_probabilities,omitempty"`
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
	IncludeExplanation  bool    `json:"include_explanation,omitempty"`
}

// EndpointInfo represents information about an API endpoint
type EndpointInfo struct {
	Path        string           `json:"path"`
	Method      string           `json:"method"`
	Description string           `json:"description"`
	Permission  RoutePermission  `json:"permission"`
	Sensitivity RouteSensitivity `json:"sensitivity"`
	EndpointContract
}

// TaskTypeInfo represents information about a task type
type TaskTypeInfo struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// EndpointMetadata stores metadata about an endpoint for API documentation
type EndpointMetadata struct {
	Path        string
	Method      string
	Description string
	Parameters  []OpenAPIParameter
	EndpointContract
}
