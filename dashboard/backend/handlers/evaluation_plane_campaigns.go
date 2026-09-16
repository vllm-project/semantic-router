package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

type evaluationControlledPairWire struct {
	BaselineRunID  evaluationJSONField[string] `json:"baseline_run_id"`
	CandidateRunID evaluationJSONField[string] `json:"candidate_run_id"`
}

func (wire *evaluationControlledPairWire) UnmarshalJSON(data []byte) error {
	type exact evaluationControlledPairWire
	return decodeExactCampaignObject(data, (*exact)(wire), "g3_controlled_pair")
}

type evaluationFidelityPairWire struct {
	ReferenceRunID evaluationJSONField[string] `json:"reference_run_id"`
	LiveRunID      evaluationJSONField[string] `json:"live_run_id"`
}

func (wire *evaluationFidelityPairWire) UnmarshalJSON(data []byte) error {
	type exact evaluationFidelityPairWire
	return decodeExactCampaignObject(data, (*exact)(wire), "g5_fidelity")
}

type evaluationCampaignGateBindingsWire struct {
	G2RunID          evaluationJSONField[string]                       `json:"g2_run_id"`
	G3ControlledPair evaluationJSONField[evaluationControlledPairWire] `json:"g3_controlled_pair"`
	G4RunID          evaluationJSONField[string]                       `json:"g4_run_id"`
	G5Fidelity       evaluationJSONField[evaluationFidelityPairWire]   `json:"g5_fidelity"`
	G6RunID          evaluationJSONField[string]                       `json:"g6_run_id"`
	G7RunID          evaluationJSONField[string]                       `json:"g7_run_id"`
	G8RunID          evaluationJSONField[string]                       `json:"g8_run_id"`
	G9RunID          evaluationJSONField[string]                       `json:"g9_run_id"`
}

func (wire *evaluationCampaignGateBindingsWire) UnmarshalJSON(data []byte) error {
	type exact evaluationCampaignGateBindingsWire
	return decodeExactCampaignObject(data, (*exact)(wire), "gate_bindings")
}

func decodeExactCampaignObject(data []byte, destination any, name string) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("campaign %s contains trailing JSON", name)
		}
		return err
	}
	return nil
}

type evaluationCreateCampaignWireRequest struct {
	ClientRequestID evaluationJSONField[string]                             `json:"client_request_id"`
	Name            evaluationJSONField[string]                             `json:"name"`
	Description     evaluationJSONField[string]                             `json:"description"`
	ChangeProfile   evaluationJSONField[evaluationplane.ChangeProfile]      `json:"change_profile"`
	GateBindings    evaluationJSONField[evaluationCampaignGateBindingsWire] `json:"gate_bindings"`
}

func (wire evaluationCreateCampaignWireRequest) domainRequest() (evaluationplane.CreateCampaignRequest, error) {
	clientRequestID, err := requiredEvaluationJSONField("client_request_id", wire.ClientRequestID)
	if err != nil {
		return evaluationplane.CreateCampaignRequest{}, err
	}
	name, err := requiredEvaluationJSONField("name", wire.Name)
	if err != nil {
		return evaluationplane.CreateCampaignRequest{}, err
	}
	description, err := requiredEvaluationJSONField("description", wire.Description)
	if err != nil {
		return evaluationplane.CreateCampaignRequest{}, err
	}
	profile, err := requiredEvaluationJSONField("change_profile", wire.ChangeProfile)
	if err != nil {
		return evaluationplane.CreateCampaignRequest{}, err
	}
	bindingsWire, err := requiredEvaluationJSONField("gate_bindings", wire.GateBindings)
	if err != nil {
		return evaluationplane.CreateCampaignRequest{}, err
	}
	bindings, err := bindingsWire.domainBindings()
	if err != nil {
		return evaluationplane.CreateCampaignRequest{}, err
	}
	return evaluationplane.CreateCampaignRequest{
		ClientRequestID: clientRequestID, Name: name, Description: description,
		ChangeProfile: profile, GateBindings: bindings,
	}, nil
}

func (wire evaluationCampaignGateBindingsWire) domainBindings() (evaluationplane.CampaignGateBindings, error) {
	result := evaluationplane.CampaignGateBindings{}
	stringsToBind := []struct {
		name  string
		field evaluationJSONField[string]
		set   func(string)
	}{
		{"gate_bindings.g2_run_id", wire.G2RunID, func(value string) { result.G2RunID = value }},
		{"gate_bindings.g4_run_id", wire.G4RunID, func(value string) { result.G4RunID = value }},
		{"gate_bindings.g6_run_id", wire.G6RunID, func(value string) { result.G6RunID = value }},
		{"gate_bindings.g7_run_id", wire.G7RunID, func(value string) { result.G7RunID = value }},
		{"gate_bindings.g8_run_id", wire.G8RunID, func(value string) { result.G8RunID = value }},
		{"gate_bindings.g9_run_id", wire.G9RunID, func(value string) { result.G9RunID = value }},
	}
	for _, binding := range stringsToBind {
		if binding.field.null {
			return result, fmt.Errorf("%s cannot be null", binding.name)
		}
		if binding.field.present {
			if binding.field.value == "" {
				return result, fmt.Errorf("%s cannot be empty", binding.name)
			}
			binding.set(binding.field.value)
		}
	}
	if wire.G3ControlledPair.null {
		return result, fmt.Errorf("gate_bindings.g3_controlled_pair cannot be null")
	}
	if wire.G3ControlledPair.present {
		pair := wire.G3ControlledPair.value
		baseline, err := requiredEvaluationJSONField("gate_bindings.g3_controlled_pair.baseline_run_id", pair.BaselineRunID)
		if err != nil {
			return result, err
		}
		candidate, err := requiredEvaluationJSONField("gate_bindings.g3_controlled_pair.candidate_run_id", pair.CandidateRunID)
		if err != nil {
			return result, err
		}
		result.G3ControlledPair = &evaluationplane.CampaignControlledPairBinding{
			BaselineRunID: baseline, CandidateRunID: candidate,
		}
	}
	if wire.G5Fidelity.null {
		return result, fmt.Errorf("gate_bindings.g5_fidelity cannot be null")
	}
	if wire.G5Fidelity.present {
		pair := wire.G5Fidelity.value
		reference, err := requiredEvaluationJSONField("gate_bindings.g5_fidelity.reference_run_id", pair.ReferenceRunID)
		if err != nil {
			return result, err
		}
		live, err := requiredEvaluationJSONField("gate_bindings.g5_fidelity.live_run_id", pair.LiveRunID)
		if err != nil {
			return result, err
		}
		result.G5Fidelity = &evaluationplane.CampaignFidelityBinding{ReferenceRunID: reference, LiveRunID: live}
	}
	return result, nil
}

func (h *EvaluationPlaneHandler) Campaigns(w http.ResponseWriter, r *http.Request) {
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
	var wire evaluationCreateCampaignWireRequest
	if err := decodeStrictJSON(r, &wire); err != nil {
		writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
		return
	}
	request, err := wire.domainRequest()
	if err != nil {
		writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
		return
	}
	campaign, err := h.service.CreateCampaignAs(actor, request)
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	writeEvaluationJSON(w, http.StatusCreated, campaign)
}

func (h *EvaluationPlaneHandler) CampaignRoute(w http.ResponseWriter, r *http.Request) {
	if evaluationCORS(w, r) {
		return
	}
	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, evaluationAPIBase+"/campaigns/"), "/")
	parts := strings.Split(rest, "/")
	if len(parts) == 0 || parts[0] == "" || len(parts) > 2 ||
		(len(parts) == 2 && parts[1] != "decision" && parts[1] != "lifecycle") {
		http.NotFound(w, r)
		return
	}
	if len(parts) == 2 && parts[1] == "lifecycle" {
		h.campaignLifecycle(w, r, parts[0])
		return
	}
	if len(parts) == 2 && r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	if len(parts) == 1 && r.Method != http.MethodGet && r.Method != http.MethodDelete {
		methodNotAllowed(w, http.MethodGet, http.MethodDelete)
		return
	}
	actor, ok := h.evaluationActor(w, r)
	if !ok {
		return
	}
	if r.Method == http.MethodDelete {
		if h.denyReadonly(w) {
			return
		}
		if err := h.service.DeleteCampaignAs(actor, parts[0]); err != nil {
			writeEvaluationError(w, err)
			return
		}
		w.WriteHeader(http.StatusNoContent)
		return
	}
	campaign, err := h.service.GetCampaignAs(actor, parts[0])
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	if len(parts) == 2 {
		writeEvaluationJSON(w, http.StatusOK, campaign.Decision)
		return
	}
	writeEvaluationJSON(w, http.StatusOK, campaign)
}
