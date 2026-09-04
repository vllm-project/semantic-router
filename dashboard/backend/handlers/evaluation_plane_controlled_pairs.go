package handlers

import (
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

// The controlled-pair wire request deliberately contains only durable UUIDs.
// Endpoint origins, credentials, labels, and version claims are resolved from
// server-sealed source manifests and cannot be supplied by a browser.
type evaluationCreateControlledPairWireRequest struct {
	ClientRequestID      evaluationJSONField[string] `json:"client_request_id"`
	BaselineSourceRunID  evaluationJSONField[string] `json:"baseline_source_run_id"`
	CandidateSourceRunID evaluationJSONField[string] `json:"candidate_source_run_id"`
	BaselineRunID        evaluationJSONField[string] `json:"baseline_run_id"`
	CandidateRunID       evaluationJSONField[string] `json:"candidate_run_id"`
}

func (wire evaluationCreateControlledPairWireRequest) domainRequest() (evaluationplane.CreateControlledPairRequest, error) {
	clientRequestID, err := requiredEvaluationJSONField("client_request_id", wire.ClientRequestID)
	if err != nil {
		return evaluationplane.CreateControlledPairRequest{}, err
	}
	baselineSource, err := requiredEvaluationJSONField("baseline_source_run_id", wire.BaselineSourceRunID)
	if err != nil {
		return evaluationplane.CreateControlledPairRequest{}, err
	}
	candidateSource, err := requiredEvaluationJSONField("candidate_source_run_id", wire.CandidateSourceRunID)
	if err != nil {
		return evaluationplane.CreateControlledPairRequest{}, err
	}
	baselineRun, err := requiredEvaluationJSONField("baseline_run_id", wire.BaselineRunID)
	if err != nil {
		return evaluationplane.CreateControlledPairRequest{}, err
	}
	candidateRun, err := requiredEvaluationJSONField("candidate_run_id", wire.CandidateRunID)
	if err != nil {
		return evaluationplane.CreateControlledPairRequest{}, err
	}
	return evaluationplane.CreateControlledPairRequest{
		ClientRequestID: clientRequestID, BaselineSourceRunID: baselineSource,
		CandidateSourceRunID: candidateSource, BaselineRunID: baselineRun,
		CandidateRunID: candidateRun,
	}, nil
}

func (h *EvaluationPlaneHandler) ControlledPairs(w http.ResponseWriter, r *http.Request) {
	if evaluationCORS(w, r) {
		return
	}
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	if r.URL.Path != evaluationAPIBase+"/controlled-pairs" || r.URL.RawQuery != "" {
		writeEvaluationError(w, fmt.Errorf("%w: controlled pair create path does not accept query parameters", evaluationplane.ErrInvalid))
		return
	}
	actor, ok := h.evaluationActor(w, r)
	if !ok {
		return
	}
	if h.denyReadonly(w) {
		return
	}
	var wire evaluationCreateControlledPairWireRequest
	if err := decodeStrictJSON(r, &wire); err != nil {
		writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
		return
	}
	request, err := wire.domainRequest()
	if err != nil {
		writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
		return
	}
	execution, err := h.service.CreateControlledPairExecutionAs(r.Context(), actor, request)
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	writeEvaluationJSON(w, http.StatusCreated, execution)
}
