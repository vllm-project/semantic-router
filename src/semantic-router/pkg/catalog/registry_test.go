package catalog

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestBuiltInDigestMatchesPublishedCatalog(t *testing.T) {
	payload, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "website", "static", "model-catalog", "catalog.json"))
	if err != nil {
		t.Fatalf("read published catalog: %v", err)
	}
	sum := sha256.Sum256(payload)
	want := "sha256:" + hex.EncodeToString(sum[:])

	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	if got := registry.Digest(); got != want {
		t.Fatalf("built-in digest = %s, want %s", got, want)
	}
	effective, err := registry.Compile(CompileInput{})
	if err != nil {
		t.Fatal(err)
	}
	if got := effective.Digest(); got != want {
		t.Fatalf("effective digest = %s, want %s", got, want)
	}
}
