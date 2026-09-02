package handlers

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

type evaluationLifecycleUpdateWireRequest struct {
	RetentionClass evaluationJSONField[evaluationplane.RetentionClass] `json:"retention_class"`
	EvidenceHold   evaluationJSONField[bool]                           `json:"evidence_hold"`
}

func (wire evaluationLifecycleUpdateWireRequest) domainRequest() (evaluationplane.UpdateLifecycleRequest, error) {
	request := evaluationplane.UpdateLifecycleRequest{}
	if wire.RetentionClass.present {
		if wire.RetentionClass.null {
			return request, fmt.Errorf("retention_class cannot be null")
		}
		request.RetentionClass = &wire.RetentionClass.value
	}
	if wire.EvidenceHold.present {
		if wire.EvidenceHold.null {
			return request, fmt.Errorf("evidence_hold cannot be null")
		}
		request.EvidenceHold = &wire.EvidenceHold.value
	}
	if request.RetentionClass == nil && request.EvidenceHold == nil {
		return request, fmt.Errorf("at least one lifecycle mutation is required")
	}
	return request, nil
}

type evaluationCollectionWireRequest struct {
	Apply      evaluationJSONField[bool]   `json:"apply"`
	PlanDigest evaluationJSONField[string] `json:"plan_digest"`
}

func (wire evaluationCollectionWireRequest) domainRequest() (evaluationplane.CollectionRequest, error) {
	apply, err := requiredEvaluationJSONField("apply", wire.Apply)
	if err != nil {
		return evaluationplane.CollectionRequest{}, err
	}
	request := evaluationplane.CollectionRequest{Apply: apply}
	if wire.PlanDigest.present {
		if wire.PlanDigest.null {
			return request, fmt.Errorf("plan_digest cannot be null")
		}
		request.PlanDigest = strings.TrimSpace(wire.PlanDigest.value)
	}
	return request, nil
}

func (h *EvaluationPlaneHandler) runLifecycle(w http.ResponseWriter, r *http.Request, runID string) {
	if evaluationCORS(w, r) {
		return
	}
	actor, ok := h.evaluationActor(w, r)
	if !ok {
		return
	}
	switch r.Method {
	case http.MethodGet:
		lifecycle, err := h.service.RunLifecycle(actor, runID)
		if err != nil {
			writeEvaluationError(w, err)
			return
		}
		writeEvaluationJSON(w, http.StatusOK, lifecycle)
	case http.MethodPost:
		if h.denyReadonly(w) {
			return
		}
		var wire evaluationLifecycleUpdateWireRequest
		if err := decodeStrictJSON(r, &wire); err != nil {
			writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
			return
		}
		request, err := wire.domainRequest()
		if err != nil {
			writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
			return
		}
		lifecycle, err := h.service.UpdateRunLifecycle(actor, runID, request)
		if err != nil {
			writeEvaluationError(w, err)
			return
		}
		writeEvaluationJSON(w, http.StatusOK, lifecycle)
	default:
		methodNotAllowed(w, http.MethodGet, http.MethodPost)
	}
}

func (h *EvaluationPlaneHandler) campaignLifecycle(w http.ResponseWriter, r *http.Request, campaignID string) {
	actor, ok := h.evaluationActor(w, r)
	if !ok {
		return
	}
	switch r.Method {
	case http.MethodGet:
		lifecycle, err := h.service.CampaignLifecycle(actor, campaignID)
		if err != nil {
			writeEvaluationError(w, err)
			return
		}
		writeEvaluationJSON(w, http.StatusOK, lifecycle)
	case http.MethodPost:
		if h.denyReadonly(w) {
			return
		}
		var wire evaluationLifecycleUpdateWireRequest
		if err := decodeStrictJSON(r, &wire); err != nil {
			writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
			return
		}
		request, err := wire.domainRequest()
		if err != nil {
			writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
			return
		}
		lifecycle, err := h.service.UpdateCampaignLifecycle(actor, campaignID, request)
		if err != nil {
			writeEvaluationError(w, err)
			return
		}
		writeEvaluationJSON(w, http.StatusOK, lifecycle)
	default:
		methodNotAllowed(w, http.MethodGet, http.MethodPost)
	}
}

func (h *EvaluationPlaneHandler) LifecycleUsage(w http.ResponseWriter, r *http.Request) {
	if preflightOrMethod(w, r, http.MethodGet) {
		return
	}
	actor, ok := h.evaluationActor(w, r)
	if !ok {
		return
	}
	usage, err := h.service.LifecycleUsage(actor)
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	writeEvaluationJSON(w, http.StatusOK, usage)
}

func (h *EvaluationPlaneHandler) LifecycleCollection(w http.ResponseWriter, r *http.Request) {
	if evaluationCORS(w, r) {
		return
	}
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	actor, ok := h.evaluationActor(w, r)
	if !ok {
		return
	}
	if h.denyReadonly(w) {
		return
	}
	var wire evaluationCollectionWireRequest
	if err := decodeStrictJSON(r, &wire); err != nil {
		writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
		return
	}
	request, err := wire.domainRequest()
	if err != nil {
		writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
		return
	}
	result, err := h.service.CollectLifecycle(actor, request)
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	writeEvaluationJSON(w, http.StatusOK, result)
}
