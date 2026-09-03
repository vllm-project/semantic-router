package handlers

import (
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

type evaluationCampaignReadinessWireRequest struct {
	ChangeProfile               evaluationJSONField[evaluationplane.ChangeProfile] `json:"change_profile"`
	Limit                       evaluationJSONField[int]                           `json:"limit"`
	Cursor                      evaluationJSONField[string]                        `json:"cursor"`
	ControlledPairBaselineRunID evaluationJSONField[string]                        `json:"controlled_pair_baseline_run_id"`
	FidelityReferenceRunID      evaluationJSONField[string]                        `json:"fidelity_reference_run_id"`
}

func (wire evaluationCampaignReadinessWireRequest) domainRequest() (evaluationplane.CampaignReadinessRequest, error) {
	profile, err := requiredEvaluationJSONField("change_profile", wire.ChangeProfile)
	if err != nil {
		return evaluationplane.CampaignReadinessRequest{}, err
	}
	request := evaluationplane.CampaignReadinessRequest{ChangeProfile: profile}
	for _, optional := range []struct {
		name  string
		field evaluationJSONField[string]
		set   func(string)
	}{
		{name: "cursor", field: wire.Cursor, set: func(value string) { request.Cursor = value }},
		{
			name: "controlled_pair_baseline_run_id", field: wire.ControlledPairBaselineRunID,
			set: func(value string) { request.BaselineSourceRunID = value },
		},
		{
			name: "fidelity_reference_run_id", field: wire.FidelityReferenceRunID,
			set: func(value string) { request.ReferenceRunID = value },
		},
	} {
		if optional.field.null {
			return evaluationplane.CampaignReadinessRequest{}, fmt.Errorf("%s cannot be null", optional.name)
		}
		if optional.field.present {
			if optional.name != "cursor" && optional.field.value == "" {
				return evaluationplane.CampaignReadinessRequest{}, fmt.Errorf("%s cannot be empty", optional.name)
			}
			optional.set(optional.field.value)
		}
	}
	if wire.Limit.null {
		return evaluationplane.CampaignReadinessRequest{}, fmt.Errorf("limit cannot be null")
	}
	if wire.Limit.present {
		request.Limit = wire.Limit.value
	}
	return request, nil
}

func (h *EvaluationPlaneHandler) CampaignReadiness(w http.ResponseWriter, r *http.Request) {
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
	var wire evaluationCampaignReadinessWireRequest
	if err := decodeStrictJSON(r, &wire); err != nil {
		writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
		return
	}
	request, err := wire.domainRequest()
	if err != nil {
		writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
		return
	}
	readiness, err := h.service.CampaignReadinessAs(actor, request)
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	writeEvaluationJSON(w, http.StatusOK, readiness)
}
