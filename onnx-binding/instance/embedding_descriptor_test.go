//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestOwnedEmbeddingDescriptorPrimaryCloneAndNoReload(t *testing.T) {
	options := layeredEmbeddingFixture(t)
	options.ModelFile = filepath.Join("onnx", "layer-1", "model.onnx")
	model, loadErr := LoadEmbeddingModel(options)
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	defer model.Close()
	read := func(m *EmbeddingModel, layer, dimension int) map[string]any {
		t.Helper()
		raw, err := m.RuntimeDescriptor(layer, dimension)
		if err != nil {
			t.Fatal(err)
		}
		var descriptor map[string]any
		if decodeErr := json.Unmarshal([]byte(raw), &descriptor); decodeErr != nil {
			t.Fatal(decodeErr)
		}
		return descriptor
	}
	original := read(model, 0, 0)
	if original["layer"] != float64(1) || original["dimension"] != float64(3) {
		t.Fatalf("default descriptor ignored actual early primary: %v", original)
	}
	if !reflect.DeepEqual(original, read(model, 1, 3)) {
		t.Fatal("default and explicit primary identities differ")
	}
	if selected := read(model, 2, 2); selected["layer"] != float64(2) || selected["dimension"] != float64(2) {
		t.Fatalf("wrong explicit exit: %v", selected)
	}
	for _, args := range [][2]int{{-1, 0}, {0, -1}, {3, 3}, {1, 4}} {
		if _, err := model.RuntimeDescriptor(args[0], args[1]); err == nil {
			t.Fatalf("accepted invalid descriptor arguments %v", args)
		}
	}
	info, err := model.Info()
	if err != nil || info.CompletedInferences != 0 {
		t.Fatalf("descriptor performed inference: %+v, %v", info, err)
	}
	if len(info.Sessions) != 2 {
		t.Fatalf("missing loaded session evidence: %+v", info.Sessions)
	}
	for _, session := range info.Sessions {
		if session.CompilationCache != nil || session.CompilerFlags == nil || len(session.CompilerFlags) != 0 {
			t.Fatalf("CPU session must expose an empty compiler snapshot without a cache: %+v", session)
		}
	}
	clone, err := model.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	// Metadata belongs to the successfully loaded sessions, not these paths.
	moved := options.ModelPath + "-moved"
	if err = os.Rename(options.ModelPath, moved); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(moved) })
	if !reflect.DeepEqual(original, read(model, 0, 0)) {
		t.Fatal("descriptor reopened source artifacts")
	}
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}
	_, err = model.RuntimeDescriptor(0, 0)
	errorKind(t, err, "closed")
	if !reflect.DeepEqual(original, read(clone, 0, 0)) {
		t.Fatal("closing original invalidated clone descriptor")
	}
}

func TestOwnedEmbeddingDescriptorRejectsWrongTask(t *testing.T) {
	sequence, err := LoadSequenceClassifier(fixture("sequence"))
	if err != nil {
		t.Fatal(err)
	}
	defer sequence.Close()
	if _, err = (&EmbeddingModel{owner: sequence.owner}).RuntimeDescriptor(0, 0); err == nil {
		t.Fatal("sequence instance claimed an embedding descriptor")
	}
}
