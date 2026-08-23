package config

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

func TestParseYAMLBytesRecordsExactDocumentHash(t *testing.T) {
	document := []byte(entrypointRulesYAML)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes(document)
	if err != nil {
		t.Fatalf("testAuthoringParser(t).ParseYAMLBytes() error = %v", err)
	}

	digest := sha256.Sum256(document)
	want := hex.EncodeToString(digest[:])
	if cfg.DocumentHash != want {
		t.Fatalf("DocumentHash = %q, want %q", cfg.DocumentHash, want)
	}
}

func TestParseYAMLBytesDocumentHashTracksFormattingChanges(t *testing.T) {
	first, err := testAuthoringParser(t).ParseYAMLBytes([]byte(entrypointRulesYAML))
	if err != nil {
		t.Fatalf("testAuthoringParser(t).ParseYAMLBytes(first) error = %v", err)
	}
	second, err := testAuthoringParser(t).ParseYAMLBytes([]byte(entrypointRulesYAML + "\n"))
	if err != nil {
		t.Fatalf("testAuthoringParser(t).ParseYAMLBytes(second) error = %v", err)
	}
	if first.DocumentHash == second.DocumentHash {
		t.Fatal("document hash should identify the exact runtime file, including formatting")
	}
}
