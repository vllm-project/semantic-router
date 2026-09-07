package recipe

import (
	"archive/zip"
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func recipeZIPFromDirectory(t *testing.T, directory string, mutate func(*zip.Writer)) []byte {
	t.Helper()
	var buffer bytes.Buffer
	writer := zip.NewWriter(&buffer)
	for _, spec := range recipeFiles {
		data, err := os.ReadFile(filepath.Join(directory, spec.name))
		if err != nil {
			t.Fatalf("ReadFile(%s): %v", spec.name, err)
		}
		entry, err := writer.Create(spec.name)
		if err != nil {
			t.Fatalf("Create(%s): %v", spec.name, err)
		}
		if _, err := entry.Write(data); err != nil {
			t.Fatalf("Write(%s): %v", spec.name, err)
		}
	}
	if mutate != nil {
		mutate(writer)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("Close ZIP: %v", err)
	}
	return buffer.Bytes()
}

func copyMaintainedRecipe(t *testing.T, name string) string {
	t.Helper()
	source := filepath.Join("..", "..", "..", "config", "recipes", name)
	target := t.TempDir()
	for _, spec := range recipeFiles {
		input, err := os.Open(filepath.Join(source, spec.name))
		if err != nil {
			t.Fatalf("Open(%s): %v", spec.name, err)
		}
		data, err := io.ReadAll(input)
		_ = input.Close()
		if err != nil {
			t.Fatalf("Read(%s): %v", spec.name, err)
		}
		if err := os.WriteFile(filepath.Join(target, spec.name), data, 0o600); err != nil {
			t.Fatalf("WriteFile(%s): %v", spec.name, err)
		}
	}
	return target
}
