package config

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"
)

func TestParseYAMLBytesRecordsExactDocumentHash(t *testing.T) {
	document := []byte(recipeTestBaseYAML)
	cfg, err := ParseYAMLBytes(document)
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}

	digest := sha256.Sum256(document)
	want := hex.EncodeToString(digest[:])
	if cfg.DocumentHash != want {
		t.Fatalf("DocumentHash = %q, want %q", cfg.DocumentHash, want)
	}
	if string(cfg.SourceDocument) != string(document) {
		t.Fatal("SourceDocument should retain the exact parsed YAML bytes")
	}
}

func TestParseYAMLBytesDocumentHashTracksFormattingChanges(t *testing.T) {
	first, err := ParseYAMLBytes([]byte(recipeTestBaseYAML))
	if err != nil {
		t.Fatalf("ParseYAMLBytes(first) error = %v", err)
	}
	second, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + "\n"))
	if err != nil {
		t.Fatalf("ParseYAMLBytes(second) error = %v", err)
	}
	if first.DocumentHash == second.DocumentHash {
		t.Fatal("document hash should identify the exact runtime file, including formatting")
	}
}

func TestRouterConfigJSONOmitsSourceDocument(t *testing.T) {
	const canary = "redact-me-canary-value-0001"
	cfg := &RouterConfig{
		ConfigBaseDir:  "/tmp/runtime-only",
		DocumentHash:   "runtime-hash",
		SourceDocument: []byte("auth_token: " + canary + "\n"),
	}
	encodedSource := base64.StdEncoding.EncodeToString(cfg.SourceDocument)

	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("json.Marshal(RouterConfig) error = %v", err)
	}
	body := string(data)
	if strings.Contains(body, canary) {
		t.Fatalf("RouterConfig JSON leaked SourceDocument canary: %s", body)
	}
	if strings.Contains(body, encodedSource) {
		t.Fatalf("RouterConfig JSON leaked encoded SourceDocument: %s", body)
	}
	for _, key := range []string{"SourceDocument", "sourceDocument", "DocumentHash", "documentHash", "ConfigBaseDir", "configBaseDir"} {
		if strings.Contains(body, `"`+key+`"`) {
			t.Fatalf("RouterConfig JSON included runtime-only field %q: %s", key, body)
		}
	}
}
