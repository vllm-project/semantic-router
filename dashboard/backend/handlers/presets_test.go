package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestPresetsHandler_ReturnsCatalog(t *testing.T) {
	handler := PresetsHandler()
	req := httptest.NewRequest(http.MethodGet, "/api/setup/presets", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var presets []PresetInfo
	if err := json.Unmarshal(rec.Body.Bytes(), &presets); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if len(presets) < 2 {
		t.Fatalf("expected at least 2 presets, got %d", len(presets))
	}

	ids := map[string]bool{}
	for _, p := range presets {
		ids[p.ID] = true
		if p.Label == "" {
			t.Errorf("preset %q has empty label", p.ID)
		}
		if p.Summary == "" {
			t.Errorf("preset %q has empty summary", p.ID)
		}
		if len(p.Models) == 0 {
			t.Errorf("preset %q has no required models", p.ID)
		}
		if p.RecipeURL == "" {
			t.Errorf("preset %q has empty recipe URL", p.ID)
		}
	}

	for _, expected := range []string{"balance", "security"} {
		if !ids[expected] {
			t.Errorf("missing expected preset %q", expected)
		}
	}
}

func TestPresetsHandler_RejectsNonGet(t *testing.T) {
	handler := PresetsHandler()
	req := httptest.NewRequest(http.MethodPost, "/api/setup/presets", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", rec.Code)
	}
}

func TestPresetDeltaHandler_AllMissing(t *testing.T) {
	handler := PresetDeltaHandler()
	body, _ := json.Marshal(PresetDeltaRequest{
		PresetID: "balance",
		Models:   []string{},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/setup/presets/delta", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp PresetDeltaResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.Ready {
		t.Error("expected ready=false when no models are configured")
	}
	if len(resp.ConfiguredModels) != 0 {
		t.Errorf("expected 0 configured models, got %d", len(resp.ConfiguredModels))
	}
	if len(resp.MissingModels) != 5 {
		t.Errorf("expected 5 missing models for balance preset, got %d", len(resp.MissingModels))
	}
}

func TestPresetDeltaHandler_PartialOverlap(t *testing.T) {
	handler := PresetDeltaHandler()
	body, _ := json.Marshal(PresetDeltaRequest{
		PresetID: "balance",
		Models:   []string{"qwen/qwen3.5-rocm", "openai/gpt5.4"},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/setup/presets/delta", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var resp PresetDeltaResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode: %v", err)
	}

	if resp.Ready {
		t.Error("expected ready=false with partial overlap")
	}
	if len(resp.ConfiguredModels) != 2 {
		t.Errorf("expected 2 configured, got %d", len(resp.ConfiguredModels))
	}
	if len(resp.MissingModels) != 3 {
		t.Errorf("expected 3 missing, got %d", len(resp.MissingModels))
	}
}

func TestPresetDeltaHandler_AllConfigured(t *testing.T) {
	handler := PresetDeltaHandler()
	body, _ := json.Marshal(PresetDeltaRequest{
		PresetID: "security",
		Models:   []string{"local/private-qwen", "cloud/frontier-reasoning"},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/setup/presets/delta", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var resp PresetDeltaResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode: %v", err)
	}

	if !resp.Ready {
		t.Error("expected ready=true when all models are configured")
	}
	if len(resp.MissingModels) != 0 {
		t.Errorf("expected 0 missing, got %d", len(resp.MissingModels))
	}
	if len(resp.ConfiguredModels) != 2 {
		t.Errorf("expected 2 configured, got %d", len(resp.ConfiguredModels))
	}
}

func TestPresetDeltaHandler_CaseInsensitive(t *testing.T) {
	handler := PresetDeltaHandler()
	body, _ := json.Marshal(PresetDeltaRequest{
		PresetID: "security",
		Models:   []string{"LOCAL/PRIVATE-QWEN", "Cloud/Frontier-Reasoning"},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/setup/presets/delta", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	var resp PresetDeltaResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode: %v", err)
	}

	if !resp.Ready {
		t.Error("expected case-insensitive matching to find all models")
	}
}

func TestPresetDeltaHandler_UnknownPreset(t *testing.T) {
	handler := PresetDeltaHandler()
	body, _ := json.Marshal(PresetDeltaRequest{
		PresetID: "nonexistent",
		Models:   []string{},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/setup/presets/delta", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for unknown preset, got %d", rec.Code)
	}
}

func TestPresetDeltaHandler_RejectsNonPost(t *testing.T) {
	handler := PresetDeltaHandler()
	req := httptest.NewRequest(http.MethodGet, "/api/setup/presets/delta", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", rec.Code)
	}
}

func TestComputeModelDelta_EmptyInputs(t *testing.T) {
	configured, missing := computeModelDelta([]PresetModel{}, []string{})
	if len(configured) != 0 || len(missing) != 0 {
		t.Errorf("expected empty results for empty inputs, got configured=%d missing=%d", len(configured), len(missing))
	}
}
