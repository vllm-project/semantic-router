package config

import (
	"crypto/sha256"
	"encoding/hex"
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
