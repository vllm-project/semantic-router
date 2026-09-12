//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func TestHandleConfigPlanValidatesWithoutMutation(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	candidate := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("planned_route"))
	body, err := json.Marshal(routerConfigPlanRequest{
		YAML: string(candidate),
		Mode: routerConfigMutationReplace,
	})
	if err != nil {
		t.Fatalf("marshal plan: %v", err)
	}

	request := httptest.NewRequest(http.MethodPost, apiConfigPlanPath, bytes.NewReader(body))
	response := httptest.NewRecorder()
	(&ClassificationAPIServer{configPath: configPath}).handleConfigPlan(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	var plan routerConfigPlanResponse
	if decodeErr := json.Unmarshal(response.Body.Bytes(), &plan); decodeErr != nil {
		t.Fatalf("decode plan: %v", decodeErr)
	}
	if !plan.Valid || !plan.Changed || plan.Mode != routerConfigMutationReplace {
		t.Fatalf("unexpected plan: %+v", plan)
	}
	if plan.CurrentETag != configDocumentETag(before) || plan.CandidateETag == "" {
		t.Fatalf("unexpected plan ETags: %+v", plan)
	}
	after, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config after plan: %v", err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("config plan mutated the source document")
	}
}

func TestHandleConfigPlanTreatsFormattingOnlyChangeAsUnchanged(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	body, err := json.Marshal(routerConfigPlanRequest{
		YAML: string(before),
		Mode: routerConfigMutationReplace,
	})
	if err != nil {
		t.Fatalf("marshal plan: %v", err)
	}

	request := httptest.NewRequest(http.MethodPost, apiConfigPlanPath, bytes.NewReader(body))
	response := httptest.NewRecorder()
	(&ClassificationAPIServer{configPath: configPath}).handleConfigPlan(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	var plan routerConfigPlanResponse
	if decodeErr := json.Unmarshal(response.Body.Bytes(), &plan); decodeErr != nil {
		t.Fatalf("decode plan: %v", decodeErr)
	}
	if !plan.Valid || plan.Changed {
		t.Fatalf("formatting-only plan = %+v, want valid and unchanged", plan)
	}
	if plan.CurrentETag == plan.CandidateETag {
		t.Fatalf("test fixture did not produce distinct byte identities: %+v", plan)
	}
	after, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config after plan: %v", err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("formatting-only plan mutated the source document")
	}
}

func TestHandleConfigPlanRejectsUnknownMode(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	body, err := json.Marshal(routerConfigPlanRequest{YAML: "version: v0.3\n", Mode: "patch"})
	if err != nil {
		t.Fatalf("marshal plan: %v", err)
	}
	request := httptest.NewRequest(http.MethodPost, apiConfigPlanPath, bytes.NewReader(body))
	response := httptest.NewRecorder()

	(&ClassificationAPIServer{configPath: configPath}).handleConfigPlan(response, request)

	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
}

func TestHandleConfigPlanRejectsEnvoyTopologyMutation(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	candidateConfig := minimalDeployTestConfig("old_route")
	candidateConfig.VLLMEndpoints[0].Port++
	candidate := mustMarshalCanonicalConfigYAML(t, candidateConfig)
	body, err := json.Marshal(routerConfigPlanRequest{
		YAML: string(candidate),
		Mode: routerConfigMutationReplace,
	})
	if err != nil {
		t.Fatalf("marshal plan: %v", err)
	}

	request := httptest.NewRequest(http.MethodPost, apiConfigPlanPath, bytes.NewReader(body))
	response := httptest.NewRecorder()
	(&ClassificationAPIServer{configPath: configPath}).handleConfigPlan(response, request)

	if response.Code != http.StatusConflict ||
		!strings.Contains(response.Body.String(), "RESTART_REQUIRED") {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	after, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config after plan: %v", err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("restart-required plan mutated the source document")
	}
}
