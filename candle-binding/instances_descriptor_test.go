//go:build !windows && cgo && (amd64 || arm64)

package candle_binding

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"
)

func TestOwnedEmbeddingDescriptorIdentityAndLifecycle(t *testing.T) {
	path := ownedModelFixture(t, 0)
	model, loadErr := LoadEmbeddingModel(InstanceOptions{ModelPath: path, ModelType: "mmbert"})
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
	if original["layer"] != float64(1) || original["dimension"] != float64(4) || original["model_type"] != "mmbert" {
		t.Fatalf("wrong actual representation: %v", original)
	}
	if !reflect.DeepEqual(original, read(model, 1, 4)) {
		t.Fatal("explicit full exit differs from default")
	}
	if read(model, 1, 2)["dimension"] != float64(2) {
		t.Fatal("descriptor ignored requested dimension")
	}
	for _, args := range [][2]int{{-1, 0}, {0, -1}, {2, 4}, {1, 5}} {
		if _, err := model.RuntimeDescriptor(args[0], args[1]); err == nil {
			t.Fatalf("accepted invalid descriptor arguments %v", args)
		}
	}
	other, err := LoadEmbeddingModel(InstanceOptions{ModelPath: ownedModelFixture(t, 1), ModelType: "mmbert"})
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	if reflect.DeepEqual(original["artifacts"], read(other, 0, 0)["artifacts"]) {
		t.Fatal("different loaded artifacts shared an identity")
	}
	clone, err := model.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	// The captured descriptor must not reopen the requested path.
	moved := path + "-moved"
	if err = os.Rename(path, moved); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(moved) })
	if !reflect.DeepEqual(original, read(model, 0, 0)) {
		t.Fatal("moving source files changed a live instance identity")
	}
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = model.RuntimeDescriptor(0, 0); !errors.Is(err, ErrInstanceClosed) {
		t.Fatalf("closed descriptor: %v", err)
	}
	if !reflect.DeepEqual(original, read(clone, 0, 0)) {
		t.Fatal("closing original changed clone identity")
	}
}

func TestOwnedEmbeddingDescriptorRejectsWrongTask(t *testing.T) {
	sequence, err := LoadSequenceClassifier(InstanceOptions{ModelPath: ownedModelFixture(t, 0)})
	if err != nil {
		t.Fatal(err)
	}
	defer sequence.Close()
	_, err = (&EmbeddingModel{instance: sequence.instance}).RuntimeDescriptor(0, 0)
	var boundary *InstanceError
	if !errors.As(err, &boundary) || boundary.Code != "capability" {
		t.Fatalf("wrong task descriptor: %v", err)
	}
}
