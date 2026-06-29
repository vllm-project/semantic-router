//go:build !windows && cgo

package apiserver

import (
	"context"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

type RouterOutcomeRequest struct {
	ReplayID  string            `json:"replay_id"`
	Source    string            `json:"source"`
	Target    string            `json:"target"`
	TargetRef string            `json:"target_ref,omitempty"`
	Verdict   string            `json:"verdict"`
	Reason    string            `json:"reason,omitempty"`
	Score     *float64          `json:"score,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type RouterOutcomeResponse struct {
	Success   bool   `json:"success"`
	Updated   int    `json:"updated"`
	Recorded  bool   `json:"recorded"`
	Timestamp string `json:"timestamp"`
}

type routerOutcomeValidationError struct {
	code    string
	message string
}

func (s *ClassificationAPIServer) handleRouterOutcome(w http.ResponseWriter, r *http.Request) {
	var req RouterOutcomeRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	outcome, validationErr := normalizeRouterOutcomeRequest(req)
	if validationErr != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, validationErr.code, validationErr.message)
		return
	}
	runtime := s.currentLearningRuntime()
	if runtime == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "NO_ROUTER_LEARNING_RUNTIME",
			"Router Learning outcome ingestion requires an active router learning runtime.")
		return
	}
	ctx := r.Context()
	if ctx == nil {
		ctx = context.Background()
	}
	result := runtime.UpdateOutcome(ctx, outcome)
	s.writeJSONResponse(w, http.StatusOK, RouterOutcomeResponse{
		Success:   true,
		Updated:   result.Updated,
		Recorded:  result.Recorded,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	})
}

func normalizeRouterOutcomeRequest(req RouterOutcomeRequest) (*routerruntime.RouterOutcome, *routerOutcomeValidationError) {
	replayID := strings.TrimSpace(req.ReplayID)
	if replayID == "" {
		return nil, &routerOutcomeValidationError{code: "INVALID_OUTCOME", message: "replay_id is required"}
	}
	source, ok := routerOutcomeSource(req.Source)
	if !ok {
		return nil, &routerOutcomeValidationError{code: "INVALID_OUTCOME", message: "source must be one of user, agent, eval, operator, provider, router"}
	}
	target, ok := routerOutcomeTarget(req.Target)
	if !ok {
		return nil, &routerOutcomeValidationError{code: "INVALID_OUTCOME", message: "target must be one of model, route, policy, stability, provider, router"}
	}
	verdict, ok := routerOutcomeVerdict(req.Verdict)
	if !ok {
		return nil, &routerOutcomeValidationError{code: "INVALID_OUTCOME", message: "verdict must be one of good_fit, underpowered, overprovisioned, failed"}
	}
	score := 0.0
	if req.Score != nil {
		score = *req.Score
		if math.IsNaN(score) || math.IsInf(score, 0) || score < 0 || score > 1 {
			return nil, &routerOutcomeValidationError{code: "INVALID_OUTCOME", message: "score must be between 0.0 and 1.0"}
		}
	}
	return &routerruntime.RouterOutcome{
		ReplayID:  replayID,
		Source:    source,
		Target:    target,
		TargetRef: boundedOutcomeTargetRef(req.TargetRef),
		Verdict:   verdict,
		Reason:    strings.TrimSpace(req.Reason),
		Score:     score,
		Metadata:  boundedOutcomeMetadata(req.Metadata),
	}, nil
}

func routerOutcomeSource(value string) (routerruntime.RouterOutcomeSource, bool) {
	switch routerruntime.RouterOutcomeSource(strings.TrimSpace(value)) {
	case routerruntime.RouterOutcomeSourceUser,
		routerruntime.RouterOutcomeSourceAgent,
		routerruntime.RouterOutcomeSourceEval,
		routerruntime.RouterOutcomeSourceOperator,
		routerruntime.RouterOutcomeSourceProvider,
		routerruntime.RouterOutcomeSourceRouter:
		return routerruntime.RouterOutcomeSource(strings.TrimSpace(value)), true
	default:
		return "", false
	}
}

func routerOutcomeTarget(value string) (routerruntime.RouterOutcomeTarget, bool) {
	switch routerruntime.RouterOutcomeTarget(strings.TrimSpace(value)) {
	case routerruntime.RouterOutcomeTargetModel,
		routerruntime.RouterOutcomeTargetRoute,
		routerruntime.RouterOutcomeTargetPolicy,
		routerruntime.RouterOutcomeTargetStability,
		routerruntime.RouterOutcomeTargetProvider,
		routerruntime.RouterOutcomeTargetRouter:
		return routerruntime.RouterOutcomeTarget(strings.TrimSpace(value)), true
	default:
		return "", false
	}
}

func routerOutcomeVerdict(value string) (routerruntime.RouterOutcomeVerdict, bool) {
	switch routerruntime.RouterOutcomeVerdict(strings.TrimSpace(value)) {
	case routerruntime.RouterOutcomeVerdictGoodFit,
		routerruntime.RouterOutcomeVerdictUnderpowered,
		routerruntime.RouterOutcomeVerdictOverprovisioned,
		routerruntime.RouterOutcomeVerdictFailed:
		return routerruntime.RouterOutcomeVerdict(strings.TrimSpace(value)), true
	default:
		return "", false
	}
}

func boundedOutcomeMetadata(metadata map[string]string) map[string]string {
	if len(metadata) == 0 {
		return nil
	}
	result := make(map[string]string, min(len(metadata), 32))
	count := 0
	for key, value := range metadata {
		if count >= 32 {
			break
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if key == "" || len(key) > 128 || len(value) > 1024 {
			continue
		}
		result[key] = value
		count++
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

func boundedOutcomeTargetRef(value string) string {
	value = strings.TrimSpace(value)
	if len(value) > 512 {
		return value[:512]
	}
	return value
}
