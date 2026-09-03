package recipe

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestServiceReadsActiveObjectAndFailsClosedOnRuntimeDrift(t *testing.T) {
	directory := copyMaintainedRecipe(t, "accuracy")
	archive := recipeZIPFromDirectory(t, directory, nil)
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(archive) }))
	defer server.Close()
	root := t.TempDir()
	configPath := filepath.Join(root, "runtime-config.yaml")
	previous := []byte("version: v0.3\n")
	if err := os.WriteFile(configPath, previous, 0o600); err != nil {
		t.Fatal(err)
	}
	store := NewStore(StoreOptions{Root: filepath.Join(root, "store"), ConfigPath: configPath, HTTPClient: server.Client()})
	installed, _, err := store.Import(context.Background(), ImportRequest{URL: server.URL + "/accuracy.zip"})
	if err != nil {
		t.Fatal(err)
	}
	target, rawConfig, err := store.ActivationTarget(installed.RecipeDigest, true)
	if err != nil {
		t.Fatal(err)
	}
	transaction, err := store.BeginActivation(target.RecipeDigest, previous)
	if err != nil {
		t.Fatal(err)
	}
	if writeErr := os.WriteFile(configPath, rawConfig, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	if commitErr := store.CommitActivation(transaction, ActivePointer{
		RecipeDigest:         target.RecipeDigest,
		ConfigDigest:         target.ConfigDigest,
		RealizedConfigDigest: digestBytes(rawConfig),
	}); commitErr != nil {
		t.Fatal(commitErr)
	}

	service := NewService(Options{
		Directory:         filepath.Join("..", "..", "..", "config", "recipes", "privacy"),
		Store:             store,
		RuntimeConfigPath: configPath,
	})
	descriptor, err := service.Describe()
	if err != nil || descriptor.Metadata == nil || descriptor.Metadata.ID != "accuracy" || descriptor.ActivationState != ActivationActive {
		t.Fatalf("active descriptor = %#v, %v", descriptor, err)
	}
	if writeErr := os.WriteFile(configPath, []byte("version: v0.3\n# manual drift\n"), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	descriptor, err = service.Describe()
	if err != nil || descriptor.ActivationState != ActivationInconsistent || descriptor.SourceHealth.Status != ActivationInconsistent {
		t.Fatalf("drift descriptor = %#v, %v", descriptor, err)
	}
	if _, err := service.ListProbes(ListOptions{}); !errors.Is(err, ErrActivationState) {
		t.Fatalf("ListProbes() drift error = %v", err)
	}
}
