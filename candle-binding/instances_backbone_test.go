//go:build !windows && cgo && (amd64 || arm64)

package candle_binding

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Write a genuine head-only or backbone-only safetensors artifact; no tensor
// absent from the selected part can accidentally satisfy a loader dependency.
func ownedFixturePart(t *testing.T, winner int, backbone bool) string {
	t.Helper()
	path := ownedModelFixture(t, winner)
	file := filepath.Join(path, "model.safetensors")
	original, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	size := binary.LittleEndian.Uint64(original[:8])
	type tensorEntry struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	var tensors map[string]tensorEntry
	if err = json.Unmarshal(original[8:8+size], &tensors); err != nil {
		t.Fatal(err)
	}
	selected := map[string]tensorEntry{}
	var data []byte
	for name, entry := range tensors {
		if strings.HasPrefix(name, "model.") != backbone {
			continue
		}
		payload := original[8+size:][entry.Offsets[0]:entry.Offsets[1]]
		entry.Offsets = [2]int{len(data), len(data) + len(payload)}
		data = append(data, payload...)
		selected[name] = entry
	}
	header, err := json.Marshal(selected)
	if err != nil {
		t.Fatal(err)
	}
	for len(header)%8 != 0 {
		header = append(header, ' ')
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	out = append(out, header...)
	out = append(out, data...)
	if err = os.WriteFile(file, out, 0o600); err != nil {
		t.Fatal(err)
	}
	if backbone {
		if err = os.Remove(filepath.Join(path, "tokenizer.json")); err != nil {
			t.Fatal(err)
		}
		var cfg map[string]any
		raw, readErr := os.ReadFile(filepath.Join(path, "config.json"))
		if readErr != nil {
			t.Fatal(readErr)
		}
		if err = json.Unmarshal(raw, &cfg); err != nil {
			t.Fatal(err)
		}
		delete(cfg, "id2label")
		delete(cfg, "label2id")
		raw, err = json.Marshal(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(path, "config.json"), raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return path
}

func TestOwnedNativeHeadlessBackboneOrderAndClose(t *testing.T) {
	base := ownedFixturePart(t, 0, true)
	sequencePath := ownedFixturePart(t, 1, false)
	tokenPath := ownedFixturePart(t, 1, false)
	for _, tokenFirst := range []bool{true, false} {
		backbone, err := LoadBackbone(InstanceOptions{ModelPath: base})
		if err != nil {
			t.Fatal(err)
		}
		var sequence *SequenceClassifier
		var tokens *TokenClassifier
		if tokenFirst {
			tokens, err = backbone.BindTokenHead(tokenPath)
			if err != nil {
				t.Fatal(err)
			}
			sequence, err = backbone.BindSequenceHead(sequencePath)
		} else {
			sequence, err = backbone.BindSequenceHead(sequencePath)
			if err != nil {
				t.Fatal(err)
			}
			tokens, err = backbone.BindTokenHead(tokenPath)
		}
		if err != nil {
			t.Fatal(err)
		}
		info, err := backbone.Info()
		if err != nil {
			t.Fatal(err)
		}
		if len(info.Labels) != 0 || info.Task != "backbone" {
			t.Fatal("headless model claimed a task head")
		}
		sequenceInfo, err := sequence.Info()
		if err != nil {
			t.Fatal(err)
		}
		tokenInfo, err := tokens.Info()
		if err != nil {
			t.Fatal(err)
		}
		if info.ResourceID != sequenceInfo.ResourceID || info.ResourceID != tokenInfo.ResourceID {
			t.Fatal("heads duplicated encoder")
		}
		if err = backbone.Close(); err != nil {
			t.Fatal(err)
		}
		output, err := sequence.Classify("hello world")
		if err != nil || output.Class != 1 {
			t.Fatalf("sequence after base close: %+v %v", output, err)
		}
		if err = sequence.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err = tokens.ClassifyTokens("hello world"); err != nil {
			t.Fatal(err)
		}
		if err = tokens.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
