//go:build !windows

package apiserver

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestTypedBatchAllKeepsSiblingHeadsAndErrors(t *testing.T) {
	unified := batchAllUnifiedResponse()
	items := make([]services.ClassificationInputResult, 4)
	for offset, index := range []int{0, 1, 3} {
		items[index].Unified = &classification.UnifiedBatchResults{
			IntentResults:   unified.IntentResults[offset : offset+1],
			PIIResults:      unified.PIIResults[offset : offset+1],
			SecurityResults: unified.SecurityResults[offset : offset+1],
		}
	}
	items[2].Error = &services.ClassificationItemError{Code: "UNSUPPORTED_INPUT", Message: "unsupported classification input"}
	service := typedBatchResponseFixture{
		ClassificationService: services.NewPlaceholderClassificationService(),
		response:              &services.ClassificationInputsResponse{Results: items},
	}
	api := &ClassificationAPIServer{}
	recorder := httptest.NewRecorder()
	api.handleTypedBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", nil), service, BatchClassificationRequest{
		Inputs: make([]services.ClassificationInput, 4), TaskType: "all", Options: &ClassificationOptions{ReturnProbabilities: true},
	})
	var response BatchClassificationResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if recorder.Code != http.StatusOK || len(response.Results) != 4 {
		t.Fatalf("unexpected response: %s", recorder.Body.String())
	}
	for index, result := range response.Results {
		if result.Index == nil || *result.Index != index {
			t.Fatalf("item order changed at %d: %+v", index, result)
		}
	}
	for offset, index := range []int{0, 1, 3} {
		result := response.Results[index]
		if result.Error != nil || result.Category != unified.IntentResults[offset].Category || result.PII == nil || result.Security == nil {
			t.Fatalf("successful sibling lost a head at %d: %+v", index, result)
		}
		assertSameJSON(t, "typed all pii", unified.PIIResults[offset], result.PII)
		assertSameJSON(t, "typed all security", unified.SecurityResults[offset], result.Security)
	}
	failed := response.Results[2]
	if failed.Error == nil || failed.Error.Code != "UNSUPPORTED_INPUT" || failed.PII != nil || failed.Security != nil || failed.Category != "" {
		t.Fatalf("failed item leaked a successful head: %+v", failed)
	}
	if response.Results[0].Probabilities["economics"] != .5 || response.Statistics.AvgConfidence != .5 || response.Statistics.LowConfidenceCount != 2 || len(response.Statistics.CategoryDistribution) != 3 {
		t.Fatalf("failed item shifted projections or statistics: %+v", response)
	}
}

func TestLegacyBatchAllClearsFailedHeads(t *testing.T) {
	unified := batchAllUnifiedResponse()
	unified.ItemErrors = []error{nil, errors.New("private inference payload"), nil}
	response := buildBatchResponseForTask(unified, "all", nil)
	failed := response.Results[1]
	if failed.Error == nil || failed.Error.Code != "CLASSIFICATION_ERROR" || failed.Error.Message != "classification failed" || failed.PII != nil || failed.Security != nil || failed.Category != "" {
		t.Fatalf("failed item retained head data or model error: %+v", failed)
	}
	for _, index := range []int{0, 2} {
		result := response.Results[index]
		if result.Error != nil || result.Category != unified.IntentResults[index].Category || result.PII == nil || result.Security == nil {
			t.Fatalf("successful sibling shifted at %d: %+v", index, result)
		}
	}
	if len(response.Statistics.CategoryDistribution) != 2 || response.Statistics.AvgConfidence != .625 || response.Statistics.LowConfidenceCount != 1 {
		t.Fatalf("failed item entered statistics: %+v", response.Statistics)
	}
}
