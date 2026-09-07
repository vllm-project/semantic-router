package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const validCampaignRequestJSON = `{
  "client_request_id":"63f7b8f0-a839-40af-a2cf-e84800823948",
  "name":"schema campaign",
  "description":"independent release evidence",
  "change_profile":"schema_adapter",
  "gate_bindings":{
    "g4_run_id":"2650bcb4-05af-46e4-96d4-cb9ec36b393d"
  }
}`

func TestEvaluationCampaignCreateRejectsOldUnknownAndNullBindings(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := newAuthenticatedEvaluationTestHandler(service, false)
	for _, body := range []string{
		strings.Replace(validCampaignRequestJSON, `"g4_run_id":"2650bcb4-05af-46e4-96d4-cb9ec36b393d"`, `"g4_run_id":"2650bcb4-05af-46e4-96d4-cb9ec36b393d","endpoint":"https://attacker.invalid"`, 1),
		strings.Replace(validCampaignRequestJSON, `"g4_run_id":"2650bcb4-05af-46e4-96d4-cb9ec36b393d"`, `"g4_run_id":null`, 1),
		strings.Replace(validCampaignRequestJSON, `"g4_run_id":"2650bcb4-05af-46e4-96d4-cb9ec36b393d"`, `"g5_fidelity":null`, 1),
		strings.Replace(validCampaignRequestJSON, `"gate_bindings":{`, `"runs":{},"gate_bindings":{`, 1),
		strings.Replace(validCampaignRequestJSON, `"gate_bindings":{`, `"gate_bindings":null,"unused":{`, 1),
	} {
		response := httptest.NewRecorder()
		handler.Campaigns(response, httptest.NewRequest(http.MethodPost, evaluationAPIBase+"/campaigns", strings.NewReader(body)))
		if response.Code != http.StatusBadRequest {
			t.Fatalf("status=%d body=%s request=%s", response.Code, response.Body.String(), body)
		}
	}
}

func TestEvaluationCampaignWireMapsTypedIndependentSlots(t *testing.T) {
	body := `{
      "client_request_id":"63f7b8f0-a839-40af-a2cf-e84800823948",
      "name":"recipe campaign",
      "description":"typed slots",
      "change_profile":"recipe",
      "gate_bindings":{
        "g2_run_id":"11111111-1111-4111-8111-111111111111",
        "g3_controlled_pair":{
          "baseline_run_id":"22222222-2222-4222-8222-222222222222",
          "candidate_run_id":"33333333-3333-4333-8333-333333333333"
        },
        "g4_run_id":"44444444-4444-4444-8444-444444444444",
        "g5_fidelity":{
          "reference_run_id":"55555555-5555-4555-8555-555555555555",
          "live_run_id":"66666666-6666-4666-8666-666666666666"
        },
        "g7_run_id":"77777777-7777-4777-8777-777777777777"
      }
    }`
	var wire evaluationCreateCampaignWireRequest
	if err := json.Unmarshal([]byte(body), &wire); err != nil {
		t.Fatal(err)
	}
	request, err := wire.domainRequest()
	if err != nil {
		t.Fatal(err)
	}
	if request.GateBindings.G3ControlledPair == nil ||
		request.GateBindings.G3ControlledPair.CandidateRunID != "33333333-3333-4333-8333-333333333333" ||
		request.GateBindings.G5Fidelity == nil ||
		request.GateBindings.G5Fidelity.LiveRunID != "66666666-6666-4666-8666-666666666666" {
		t.Fatalf("bindings=%+v", request.GateBindings)
	}
}

func TestEvaluationCampaignWireRejectsIncompletePairObjects(t *testing.T) {
	for _, field := range []string{
		`"g3_controlled_pair":{"baseline_run_id":"22222222-2222-4222-8222-222222222222"}`,
		`"g5_fidelity":{"reference_run_id":"55555555-5555-4555-8555-555555555555"}`,
	} {
		body := strings.Replace(validCampaignRequestJSON, `"g4_run_id":"2650bcb4-05af-46e4-96d4-cb9ec36b393d"`, field, 1)
		var wire evaluationCreateCampaignWireRequest
		if err := json.Unmarshal([]byte(body), &wire); err != nil {
			t.Fatal(err)
		}
		if _, err := wire.domainRequest(); err == nil {
			t.Fatalf("incomplete pair accepted: %s", body)
		}
	}
}

func TestEvaluationCampaignRoutesCurrentContractOnly(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := newAuthenticatedEvaluationTestHandler(service, false)
	create := httptest.NewRecorder()
	handler.Campaigns(create, httptest.NewRequest(http.MethodPost, evaluationAPIBase+"/campaigns", strings.NewReader(validCampaignRequestJSON)))
	if create.Code != http.StatusNotFound {
		t.Fatalf("valid campaign envelope status=%d body=%s", create.Code, create.Body.String())
	}

	invalid := httptest.NewRecorder()
	handler.CampaignRoute(invalid, httptest.NewRequest(http.MethodGet, evaluationAPIBase+"/campaigns/not-a-uuid", nil))
	if invalid.Code != http.StatusBadRequest {
		t.Fatalf("invalid campaign id status=%d body=%s", invalid.Code, invalid.Body.String())
	}

	readonly := newAuthenticatedEvaluationTestHandler(service, true)
	denied := httptest.NewRecorder()
	readonly.Campaigns(denied, httptest.NewRequest(http.MethodPost, evaluationAPIBase+"/campaigns", strings.NewReader(validCampaignRequestJSON)))
	if denied.Code != http.StatusForbidden {
		t.Fatalf("readonly campaign create status=%d body=%s", denied.Code, denied.Body.String())
	}
}

func TestEvaluationCampaignRoutesRequireAuthenticatedActor(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := NewEvaluationPlaneHandler(service, false)
	campaignID := "63f7b8f0-a839-40af-a2cf-e84800823948"
	tests := []struct {
		name   string
		method string
		path   string
		body   string
		direct func(http.ResponseWriter, *http.Request)
	}{
		{"create", http.MethodPost, evaluationAPIBase + "/campaigns", validCampaignRequestJSON, handler.Campaigns},
		{"read", http.MethodGet, evaluationAPIBase + "/campaigns/" + campaignID, "", handler.CampaignRoute},
		{"decision", http.MethodGet, evaluationAPIBase + "/campaigns/" + campaignID + "/decision", "", handler.CampaignRoute},
		{"lifecycle", http.MethodGet, evaluationAPIBase + "/campaigns/" + campaignID + "/lifecycle", "", handler.CampaignRoute},
		{"delete", http.MethodDelete, evaluationAPIBase + "/campaigns/" + campaignID, "", handler.CampaignRoute},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			test.direct(response, httptest.NewRequest(test.method, test.path, strings.NewReader(test.body)))
			if response.Code != http.StatusUnauthorized {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}
