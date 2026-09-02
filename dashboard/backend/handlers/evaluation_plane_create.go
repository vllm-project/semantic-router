package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

// evaluationCreateRunWireRequest is the HTTP contract for a create request.
// Domain zero values cannot represent whether a required JSON member was
// omitted or explicitly set to null, so transport validation happens before a
// CreateRunRequest is constructed.
type evaluationCreateRunWireRequest struct {
	ClientRequestID      evaluationJSONField[string]                               `json:"client_request_id"`
	Name                 evaluationJSONField[string]                               `json:"name"`
	Description          evaluationJSONField[string]                               `json:"description"`
	SuiteIDs             evaluationJSONField[[]string]                             `json:"suite_ids"`
	TrackIDs             evaluationJSONField[[]evaluationplane.TrackID]            `json:"track_ids"`
	Mode                 evaluationJSONField[evaluationplane.Mode]                 `json:"mode"`
	TargetID             evaluationJSONField[string]                               `json:"target_id"`
	ChangeProfile        evaluationJSONField[evaluationplane.ChangeProfile]        `json:"change_profile"`
	SampleLimit          evaluationJSONField[int]                                  `json:"sample_limit"`
	Concurrency          evaluationJSONField[int]                                  `json:"concurrency"`
	CapacitySLO          evaluationJSONField[evaluationplane.CapacitySLO]          `json:"capacity_slo"`
	CapacityLoadProtocol evaluationJSONField[evaluationplane.CapacityLoadProtocol] `json:"capacity_load_protocol"`
	Seed                 evaluationJSONField[int64]                                `json:"seed"`
	BaselineRunID        evaluationJSONField[string]                               `json:"baseline_run_id"`
}

type evaluationJSONField[T any] struct {
	value   T
	present bool
	null    bool
}

func (field *evaluationJSONField[T]) UnmarshalJSON(data []byte) error {
	field.present = true
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		field.null = true
		return nil
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&field.value); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("field contains trailing JSON")
		}
		return err
	}
	return nil
}

func requiredEvaluationJSONField[T any](name string, field evaluationJSONField[T]) (T, error) {
	if !field.present {
		var zero T
		return zero, fmt.Errorf("%s is required", name)
	}
	if field.null {
		var zero T
		return zero, fmt.Errorf("%s cannot be null", name)
	}
	return field.value, nil
}

func (wire evaluationCreateRunWireRequest) domainRequest() (evaluationplane.CreateRunRequest, error) {
	clientRequestID, err := requiredEvaluationJSONField("client_request_id", wire.ClientRequestID)
	if err != nil {
		return evaluationplane.CreateRunRequest{}, err
	}
	name, err := requiredEvaluationJSONField("name", wire.Name)
	if err != nil {
		return evaluationplane.CreateRunRequest{}, err
	}
	description, err := requiredEvaluationJSONField("description", wire.Description)
	if err != nil {
		return evaluationplane.CreateRunRequest{}, err
	}
	suiteIDs, err := requiredEvaluationJSONField("suite_ids", wire.SuiteIDs)
	if err != nil {
		return evaluationplane.CreateRunRequest{}, err
	}
	trackIDs, err := requiredEvaluationJSONField("track_ids", wire.TrackIDs)
	if err != nil {
		return evaluationplane.CreateRunRequest{}, err
	}
	mode, err := requiredEvaluationJSONField("mode", wire.Mode)
	if err != nil {
		return evaluationplane.CreateRunRequest{}, err
	}
	targetID, err := requiredEvaluationJSONField("target_id", wire.TargetID)
	if err != nil {
		return evaluationplane.CreateRunRequest{}, err
	}
	changeProfile, err := requiredEvaluationJSONField("change_profile", wire.ChangeProfile)
	if err != nil {
		return evaluationplane.CreateRunRequest{}, err
	}
	sampleLimit, err := requiredEvaluationJSONField("sample_limit", wire.SampleLimit)
	if err != nil {
		return evaluationplane.CreateRunRequest{}, err
	}
	concurrency, err := requiredEvaluationJSONField("concurrency", wire.Concurrency)
	if err != nil {
		return evaluationplane.CreateRunRequest{}, err
	}
	seed, err := requiredEvaluationJSONField("seed", wire.Seed)
	if err != nil {
		return evaluationplane.CreateRunRequest{}, err
	}
	var capacitySLO *evaluationplane.CapacitySLO
	if wire.CapacitySLO.present {
		if wire.CapacitySLO.null {
			return evaluationplane.CreateRunRequest{}, fmt.Errorf("capacity_slo cannot be null")
		}
		capacitySLO = &wire.CapacitySLO.value
	}
	var capacityLoadProtocol *evaluationplane.CapacityLoadProtocol
	if wire.CapacityLoadProtocol.present {
		if wire.CapacityLoadProtocol.null {
			return evaluationplane.CreateRunRequest{}, fmt.Errorf("capacity_load_protocol cannot be null")
		}
		capacityLoadProtocol = &wire.CapacityLoadProtocol.value
	}

	baselineRunID := ""
	if wire.BaselineRunID.present {
		if wire.BaselineRunID.null {
			return evaluationplane.CreateRunRequest{}, fmt.Errorf("baseline_run_id cannot be null")
		}
		baselineRunID = wire.BaselineRunID.value
		parsed, err := uuid.Parse(baselineRunID)
		if err != nil || parsed.String() != baselineRunID {
			return evaluationplane.CreateRunRequest{}, fmt.Errorf("baseline_run_id must be a canonical UUID when present")
		}
	}

	return evaluationplane.CreateRunRequest{
		ClientRequestID:      clientRequestID,
		Name:                 name,
		Description:          description,
		SuiteIDs:             suiteIDs,
		TrackIDs:             trackIDs,
		Mode:                 mode,
		TargetID:             targetID,
		ChangeProfile:        changeProfile,
		SampleLimit:          sampleLimit,
		Concurrency:          concurrency,
		CapacitySLO:          capacitySLO,
		CapacityLoadProtocol: capacityLoadProtocol,
		Seed:                 seed,
		BaselineRunID:        baselineRunID,
	}, nil
}
