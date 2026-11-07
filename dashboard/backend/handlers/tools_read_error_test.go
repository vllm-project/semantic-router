package handlers

import (
	"bytes"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestToolsDBHandler_NotFoundHidesPath(t *testing.T) {
	var logged bytes.Buffer
	log.SetOutput(&logged)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	missingPath := filepath.Join(t.TempDir(), "missing_dir", "tools_db.json")
	req := httptest.NewRequest(http.MethodGet, "/api/tools-db", nil)
	response := httptest.NewRecorder()

	ToolsDBHandler(missingPath)(response, req)

	if response.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404; body: %s", response.Code, response.Body.String())
	}
	want := "Tools database not found\n"
	if response.Body.String() != want {
		t.Fatalf("body = %q, want %q", response.Body.String(), want)
	}
	if strings.Contains(response.Body.String(), missingPath) {
		t.Fatalf("body leaks the server path: %s", response.Body.String())
	}
	if !strings.Contains(logged.String(), missingPath) {
		t.Fatalf("server log should record the missing path, got: %s", logged.String())
	}
}

func TestToolsDBHandler_ReadErrorHidesPath(t *testing.T) {
	var logged bytes.Buffer
	log.SetOutput(&logged)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	// Pointing to a directory instead of a regular file causes os.ReadFile to return EISDIR,
	// triggering the 500 error branch rather than os.IsNotExist.
	dirPath := t.TempDir()
	req := httptest.NewRequest(http.MethodGet, "/api/tools-db", nil)
	response := httptest.NewRecorder()

	ToolsDBHandler(dirPath)(response, req)

	if response.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500; body: %s", response.Code, response.Body.String())
	}
	want := "Failed to read tools database\n"
	if response.Body.String() != want {
		t.Fatalf("body = %q, want %q", response.Body.String(), want)
	}
	if strings.Contains(response.Body.String(), dirPath) {
		t.Fatalf("body leaks the server path: %s", response.Body.String())
	}
	if !strings.Contains(logged.String(), dirPath) {
		t.Fatalf("server log should record the read error path, got: %s", logged.String())
	}
}

func TestToolsDBHandler_ParseErrorHidesDetails(t *testing.T) {
	var logged bytes.Buffer
	log.SetOutput(&logged)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	invalidJSONFile := filepath.Join(t.TempDir(), "invalid.json")
	if err := os.WriteFile(invalidJSONFile, []byte("{invalid json}"), 0o600); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodGet, "/api/tools-db", nil)
	response := httptest.NewRecorder()

	ToolsDBHandler(invalidJSONFile)(response, req)

	if response.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500; body: %s", response.Code, response.Body.String())
	}
	want := "Failed to parse tools database\n"
	if response.Body.String() != want {
		t.Fatalf("body = %q, want %q", response.Body.String(), want)
	}
	if !strings.Contains(logged.String(), invalidJSONFile) {
		t.Fatalf("server log should record the parse error path, got: %s", logged.String())
	}
}

func TestToolsDBHandler_MethodNotAllowed(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/tools-db", nil)
	response := httptest.NewRecorder()

	ToolsDBHandler("irrelevant")(response, req)

	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", response.Code)
	}
}

func TestToolsDBHandler_Success(t *testing.T) {
	validJSONFile := filepath.Join(t.TempDir(), "tools_db.json")
	if err := os.WriteFile(validJSONFile, []byte(`[{"name":"test_tool"}]`), 0o600); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodGet, "/api/tools-db", nil)
	response := httptest.NewRecorder()

	ToolsDBHandler(validJSONFile)(response, req)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", response.Code)
	}
	if response.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", response.Header().Get("Content-Type"))
	}
	if !strings.Contains(response.Body.String(), "test_tool") {
		t.Fatalf("body = %q, want containing test_tool", response.Body.String())
	}
}
