package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModelCatalogRoleRecommendationsDoNotSupplyAssignments(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name        string
		declaration string
		wantCount   int
	}{
		{name: "omitted"},
		{name: "empty", declaration: `,"recommended_pool":[]`},
		{name: "below assignment minimum", declaration: `,"recommended_pool":["local/example"]`, wantCount: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			payload := strings.Replace(validModelCatalogPayload(""),
				`"minimum_candidates":1,"traits":["chat"],"recommended_pool":["local/example"]`,
				`"minimum_candidates":2,"traits":["chat"]`+tc.declaration, 1)
			payload = strings.Replace(payload, `"status":"reproduced"`, `"status":"claimed"`, 1)
			payload = strings.Replace(payload, `,"verified_at":"2026-09-04","asset_sha256"`, `,"asset_sha256"`, 1)
			response := httptest.NewRecorder()
			ModelCatalogHandler(&fakeModelCatalogSource{payload: []byte(payload)}).ServeHTTP(
				response, httptest.NewRequest(http.MethodGet, "/api/models/catalog", nil))
			if response.Code != http.StatusOK {
				t.Fatalf("catalog rejected optional recommendations: status=%d body=%s", response.Code, response.Body.String())
			}
			var document modelCatalogEnvelope
			if err := json.Unmarshal(response.Body.Bytes(), &document); err != nil {
				t.Fatalf("decode catalog: %v", err)
			}
			model := document.Models[0]
			role := model.Roles[0]
			if !role.Required || role.MinimumCandidates != 2 || role.RecommendedPool == nil || len(role.RecommendedPool) != tc.wantCount {
				t.Fatalf("catalog changed the assignment contract: %+v", role)
			}
			if model.Verification.Status != "claimed" || model.Verification.VerifiedAt != "" {
				t.Fatalf("catalog invented live verification: %+v", model.Verification)
			}
		})
	}
}

func TestModelCatalogOptionalRecommendationsRetainRoleValidation(t *testing.T) {
	t.Parallel()

	for name, replacement := range map[string]string{
		"missing role name": `"name":"","required":true,"minimum_candidates":2,"traits":["chat"]`,
		"invalid minimum":   `"name":"balanced","required":true,"minimum_candidates":0,"traits":["chat"]`,
		"missing traits":    `"name":"balanced","required":true,"minimum_candidates":2,"traits":[]`,
		"malformed pool":    `"name":"balanced","required":true,"minimum_candidates":2,"traits":["chat"],"recommended_pool":"local/example"`,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			payload := strings.Replace(validModelCatalogPayload(""),
				`"name":"balanced","required":true,"minimum_candidates":1,"traits":["chat"],"recommended_pool":["local/example"]`, replacement, 1)
			response := httptest.NewRecorder()
			ModelCatalogHandler(&fakeModelCatalogSource{payload: []byte(payload)}).ServeHTTP(
				response, httptest.NewRequest(http.MethodGet, "/api/models/catalog", nil))
			if response.Code != http.StatusBadGateway {
				t.Fatalf("malformed role accepted: status=%d", response.Code)
			}
		})
	}
}
