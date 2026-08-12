//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

type contextCompressionPreviewRequest struct {
	Configuration       json.RawMessage `json:"configuration"`
	RequestBody         json.RawMessage `json:"request_body"`
	Model               string          `json:"model,omitempty"`
	ContextWindow       int             `json:"context_window,omitempty"`
	MaxOutputTokens     int             `json:"max_output_tokens,omitempty"`
	RequestedOutput     int             `json:"requested_output_tokens,omitempty"`
	TrustedScopePresent bool            `json:"trusted_scope_present,omitempty"`
}

type contextCompressionPreviewTarget struct {
	MessageIndex   int                           `json:"message_index"`
	BlockIndex     int                           `json:"block_index"`
	Kind           contextcompression.TargetKind `json:"kind"`
	Mode           contextcompression.TargetMode `json:"mode"`
	OriginalTokens int                           `json:"original_tokens"`
	TargetTokens   int                           `json:"target_tokens"`
	Score          float64                       `json:"score"`
}

type contextCompressionPreviewResponse struct {
	Valid                bool                              `json:"valid"`
	Applied              bool                              `json:"applied"`
	TokensBefore         int                               `json:"tokens_before"`
	TokensAfter          int                               `json:"tokens_after"`
	TokensSaved          int                               `json:"tokens_saved"`
	TargetRequestTokens  int                               `json:"target_request_tokens"`
	AvailableInputTokens int                               `json:"available_input_tokens"`
	ReservedOutputTokens int                               `json:"reserved_output_tokens"`
	MessagesCompressed   int                               `json:"messages_compressed"`
	BlocksCompressed     int                               `json:"blocks_compressed"`
	OmittedChunks        int                               `json:"omitted_chunks"`
	RecoveryKeys         int                               `json:"recovery_keys"`
	TokenCounterSource   string                            `json:"token_counter_source"`
	TriggerReason        string                            `json:"trigger_reason,omitempty"`
	SkipReason           contextcompression.SkipReason     `json:"skip_reason,omitempty"`
	Quality              string                            `json:"quality,omitempty"`
	FallbackReason       string                            `json:"fallback_reason,omitempty"`
	Targets              []contextCompressionPreviewTarget `json:"targets,omitempty"`
	Warnings             []string                          `json:"warnings,omitempty"`
}

type contextCompressionStatsResponse struct {
	contextcompression.Stats
	Audit []managementAuditEntry `json:"audit,omitempty"`
}

type contextCompressionRecoveryInvalidateRequest struct {
	Recipe    string `json:"recipe"`
	Decision  string `json:"decision"`
	UserID    string `json:"user_id"`
	RequestID string `json:"request_id"`
}

func (s *ClassificationAPIServer) handleContextCompressionCapabilities(
	w http.ResponseWriter,
	_ *http.Request,
) {
	service, recovery := s.currentContextCompression()
	if service == nil {
		service = contextcompression.NewService()
	}
	s.writeJSONResponse(w, http.StatusOK, service.Capabilities(recovery != nil))
}

func (s *ClassificationAPIServer) handleContextCompressionHealth(
	w http.ResponseWriter,
	r *http.Request,
) {
	service, recovery := s.currentContextCompression()
	if service == nil {
		s.writeErrorResponse(
			w,
			http.StatusServiceUnavailable,
			"COMPRESSION_UNAVAILABLE",
			"Context compression service is unavailable",
		)
		return
	}
	status := "healthy"
	recoveryStatus := "disabled"
	if recovery != nil {
		recoveryStatus = "healthy"
		if err := recovery.Health(r.Context()); err != nil {
			status = "degraded"
			recoveryStatus = "unavailable"
		}
	}
	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"status":   status,
		"recovery": recoveryStatus,
	})
}

func (s *ClassificationAPIServer) handleContextCompressionStats(
	w http.ResponseWriter,
	_ *http.Request,
) {
	service, _ := s.currentContextCompression()
	if service == nil {
		s.writeErrorResponse(
			w,
			http.StatusServiceUnavailable,
			"COMPRESSION_UNAVAILABLE",
			"Context compression statistics are unavailable",
		)
		return
	}
	s.writeJSONResponse(w, http.StatusOK, contextCompressionStatsResponse{
		Stats: service.Stats(),
		Audit: s.contextCompressionAuditEntries(),
	})
}

func (s *ClassificationAPIServer) handleContextCompressionPreview(
	w http.ResponseWriter,
	r *http.Request,
) {
	request, pluginConfig, ok := s.parseContextCompressionPreviewRequest(w, r)
	if !ok {
		return
	}
	if !pluginConfig.Enabled {
		s.writeJSONResponse(w, http.StatusOK, contextCompressionPreviewResponse{
			Valid:      true,
			SkipReason: contextcompression.SkipDisabled,
		})
		return
	}
	result, errorStatus, err := executeContextCompressionPreview(
		r.Context(),
		request,
		pluginConfig,
	)
	if err != nil {
		s.writeErrorResponse(
			w,
			errorStatus,
			"COMPRESSION_PREVIEW_FAILED",
			err.Error(),
		)
		return
	}
	response := buildContextCompressionPreviewResponse(result)
	addContextCompressionPreviewWarnings(&response, pluginConfig)
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) parseContextCompressionPreviewRequest(
	w http.ResponseWriter,
	r *http.Request,
) (
	contextCompressionPreviewRequest,
	*config.ContextCompressionPluginConfig,
	bool,
) {
	var request contextCompressionPreviewRequest
	if err := s.parseJSONRequest(r, &request); err != nil {
		s.writeJSONRequestError(w, err)
		return request, nil, false
	}
	if len(request.Configuration) == 0 || len(request.RequestBody) == 0 {
		s.writeErrorResponse(
			w,
			http.StatusBadRequest,
			"INVALID_COMPRESSION_PREVIEW",
			"configuration and request_body are required",
		)
		return request, nil, false
	}
	var pluginConfig config.ContextCompressionPluginConfig
	if err := yaml.UnmarshalStrict(request.Configuration, &pluginConfig); err != nil {
		s.writeErrorResponse(
			w,
			http.StatusBadRequest,
			"INVALID_COMPRESSION_CONFIG",
			"Invalid context compression configuration",
		)
		return request, nil, false
	}
	if err := config.ValidateContextCompressionPluginConfig(&pluginConfig); err != nil {
		s.writeErrorResponse(
			w,
			http.StatusBadRequest,
			"INVALID_COMPRESSION_CONFIG",
			err.Error(),
		)
		return request, nil, false
	}
	return request, &pluginConfig, true
}

func executeContextCompressionPreview(
	ctx context.Context,
	request contextCompressionPreviewRequest,
	pluginConfig *config.ContextCompressionPluginConfig,
) (contextcompression.ServiceResult, int, error) {
	var requestBody map[string]interface{}
	if err := json.Unmarshal(request.RequestBody, &requestBody); err != nil {
		return contextcompression.ServiceResult{}, http.StatusBadRequest, fmt.Errorf(
			"request_body must be a JSON object",
		)
	}
	previewStore := newPreviewRecoveryStore()
	scope := ""
	if request.TrustedScopePresent {
		scope = "preview-scope"
	}
	policy := contextcompression.PolicyFromConfig(pluginConfig, nil)
	if policy.Scoring.Method != config.ContextCompressionScoringBM25 {
		policy.Scoring.Method = config.ContextCompressionScoringBM25
	}
	result := contextcompression.NewService().Apply(
		ctx,
		contextcompression.Request{
			Model:   request.Model,
			Scope:   scope,
			Request: contextcompression.ParseRequestIR(requestBody, contextcompression.Provenance{}),
			Policy:  policy,
			Capabilities: contextcompression.ModelContextCapabilities{
				ContextWindow:   request.ContextWindow,
				MaxOutputTokens: request.MaxOutputTokens,
				RequestedOutput: request.RequestedOutput,
			},
			TokenCounter: contextcompression.HeuristicTokenCounter{},
			Recovery:     previewStore,
		},
	)
	if result.Failure != nil {
		return result, http.StatusUnprocessableEntity, result.Failure
	}
	return result, 0, nil
}

func addContextCompressionPreviewWarnings(
	response *contextCompressionPreviewResponse,
	pluginConfig *config.ContextCompressionPluginConfig,
) {
	if pluginConfig.Scoring != nil &&
		pluginConfig.Scoring.Method != config.ContextCompressionScoringBM25 {
		response.Warnings = append(
			response.Warnings,
			"preview uses BM25 because runtime embedding scoring is not invoked by the management API",
		)
	}
}

func (s *ClassificationAPIServer) handleContextCompressionRecoveryInvalidate(
	w http.ResponseWriter,
	r *http.Request,
) {
	_, recovery := s.currentContextCompression()
	if recovery == nil {
		s.writeErrorResponse(
			w,
			http.StatusServiceUnavailable,
			"COMPRESSION_RECOVERY_UNAVAILABLE",
			"Context compression recovery store is unavailable",
		)
		return
	}
	var request contextCompressionRecoveryInvalidateRequest
	if err := s.parseJSONRequest(r, &request); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	request.Recipe = strings.TrimSpace(request.Recipe)
	request.Decision = strings.TrimSpace(request.Decision)
	request.UserID = strings.TrimSpace(request.UserID)
	request.RequestID = strings.TrimSpace(request.RequestID)
	if request.Recipe == "" ||
		request.Decision == "" ||
		request.UserID == "" ||
		request.RequestID == "" {
		s.writeErrorResponse(
			w,
			http.StatusBadRequest,
			"INVALID_RECOVERY_SCOPE",
			"recipe, decision, user_id, and request_id are required",
		)
		return
	}
	scope := cache.CombineFingerprints(
		request.Recipe,
		request.Decision,
		cache.UserScopeNamespace(request.UserID),
		request.RequestID,
	)
	deleted, err := recovery.InvalidateScope(r.Context(), scope)
	if err != nil {
		s.writeErrorResponse(
			w,
			http.StatusBadGateway,
			"COMPRESSION_RECOVERY_INVALIDATE_FAILED",
			"Unable to invalidate context compression recovery scope",
		)
		return
	}
	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"deleted": deleted,
	})
}

func buildContextCompressionPreviewResponse(
	result contextcompression.ServiceResult,
) contextCompressionPreviewResponse {
	targets := make([]contextCompressionPreviewTarget, 0, len(result.Plan.Targets))
	for _, target := range result.Plan.Targets {
		targets = append(targets, contextCompressionPreviewTarget{
			MessageIndex:   target.MessageIndex,
			BlockIndex:     target.BlockIndex,
			Kind:           target.Kind,
			Mode:           target.Mode,
			OriginalTokens: target.OriginalTokens,
			TargetTokens:   target.TargetTokens,
			Score:          target.Score,
		})
	}
	return contextCompressionPreviewResponse{
		Valid:                true,
		Applied:              result.Applied,
		TokensBefore:         result.TokensBefore,
		TokensAfter:          result.TokensAfter,
		TokensSaved:          max(0, result.TokensBefore-result.TokensAfter),
		TargetRequestTokens:  result.Plan.TargetRequestTokens,
		AvailableInputTokens: result.Plan.AvailableInputTokens,
		ReservedOutputTokens: result.Plan.ReservedOutputTokens,
		MessagesCompressed:   result.MessagesCompressed,
		BlocksCompressed:     result.BlocksCompressed,
		OmittedChunks:        result.OmittedChunks,
		RecoveryKeys:         len(result.RecoveryKeys),
		TokenCounterSource:   result.Plan.TokenCounterSource,
		TriggerReason:        result.Plan.TriggerReason,
		SkipReason:           result.Plan.SkipReason,
		Quality:              result.Plan.Quality,
		FallbackReason:       result.Plan.FallbackReason,
		Targets:              targets,
	}
}

type previewRecoveryStore struct {
	entries map[string]contextcompression.RecoveryEntry
}

func newPreviewRecoveryStore() *previewRecoveryStore {
	return &previewRecoveryStore{
		entries: make(map[string]contextcompression.RecoveryEntry),
	}
}

func (store *previewRecoveryStore) Put(
	_ context.Context,
	entry contextcompression.RecoveryEntry,
	_ time.Duration,
) error {
	store.entries[entry.Scope+":"+entry.Key] = entry
	return nil
}

func (store *previewRecoveryStore) Get(
	_ context.Context,
	scope string,
	key string,
) (contextcompression.RecoveryEntry, error) {
	entry, ok := store.entries[scope+":"+key]
	if !ok {
		return contextcompression.RecoveryEntry{}, contextcompression.ErrRecoveryNotFound
	}
	return entry, nil
}

func (store *previewRecoveryStore) Health(context.Context) error { return nil }
func (store *previewRecoveryStore) Close() error                 { return nil }
func (store *previewRecoveryStore) InvalidateScope(
	_ context.Context,
	scope string,
) (int64, error) {
	var count int64
	for key, entry := range store.entries {
		if entry.Scope == scope {
			delete(store.entries, key)
			count++
		}
	}
	return count, nil
}
