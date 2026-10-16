package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestModelCatalogHandlerPreservesGeneratedModelMetadata(t *testing.T) {
	t.Parallel()

	repositoryRoot := filepath.Clean(filepath.Join(packageWorkingDirectory(t), "..", "..", ".."))
	payload, err := os.ReadFile(filepath.Join(repositoryRoot, "website", "static", "model-catalog", "catalog.json"))
	if err != nil {
		t.Fatalf("read generated public catalog: %v", err)
	}
	response := httptest.NewRecorder()
	ModelCatalogHandler(&fakeModelCatalogSource{payload: payload}).ServeHTTP(
		response, httptest.NewRequest(http.MethodGet, "/api/models/catalog", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("catalog request failed: status=%d body=%s", response.Code, response.Body.String())
	}

	// Decode the wire documents without the Go catalog types: decoding back
	// into those types would hide invented empty fields, which the frontend
	// distinguishes from absent optional metadata.
	type wireCatalog struct {
		Models []map[string]any `json:"models"`
	}
	var source, result wireCatalog
	if err := json.Unmarshal(payload, &source); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(source.Models) != len(result.Models) {
		t.Fatalf("model count changed: got %d, want %d", len(result.Models), len(source.Models))
	}
	var datedPhysical, undatedVirtual int
	for index, want := range source.Models {
		got := result.Models[index]
		if !reflect.DeepEqual(got, want) {
			t.Errorf("model %s metadata changed in the HTTP response:\ngot  %v\nwant %v", want["id"], got, want)
		}
		verification := want["verification"].(map[string]any)
		_, hasDate := verification["verified_at"]
		_, hasLimits := want["limits"]
		if want["kind"] == "physical" && hasDate && hasLimits {
			datedPhysical++
		}
		if want["kind"] == "virtual" && !hasDate && !hasLimits {
			undatedVirtual++
		}
	}
	if datedPhysical == 0 || undatedVirtual == 0 {
		t.Fatalf("fixture must cover physical limits/dates and omitted virtual metadata: physical=%d virtual=%d", datedPhysical, undatedVirtual)
	}
}
