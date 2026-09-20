//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"image"
	"image/png"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func omniFixture(t *testing.T) Options {
	t.Helper()
	source := filepath.Join("testdata", "omni")
	root := t.TempDir()
	err := filepath.WalkDir(source, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		target := filepath.Join(root, rel)
		if entry.IsDir() {
			return os.MkdirAll(target, 0700)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, 0600)
	})
	if err != nil {
		t.Fatal(err)
	}
	return Options{ModelPath: root, Provider: "cpu", IntraThreads: 1}
}
func changeOmniManifest(t *testing.T, options Options, change func(map[string]any)) {
	t.Helper()
	path := filepath.Join(options.ModelPath, "vela_omni_manifest.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var value map[string]any
	if err = json.Unmarshal(raw, &value); err != nil {
		t.Fatal(err)
	}
	change(value)
	raw, err = json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
}
func TestOmniOwnedModalitiesIdentityAndInputContracts(t *testing.T) {
	options := omniFixture(t)
	options.MaxInputTokens = 8
	model, err := LoadOmni(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	info, err := model.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.Task != "omni_embedding" || !reflect.DeepEqual(info.Modalities, []string{"text", "image", "audio"}) || !reflect.DeepEqual(info.AvailableDimensions, []int{384}) || len(info.Sessions) != 4 || info.Audio == nil || info.Audio.Layout != "channels_first" || info.EffectiveLimit != 8 || len(info.AvailableLayers) != 0 {
		t.Fatalf("incorrect loaded capabilities: %+v", info)
	}
	raw, err := model.RuntimeDescriptor(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	clone, err := model.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	for _, dimension := range []int{0, 384} {
		result, callErr := model.EncodeText("hello", dimension)
		if callErr != nil || len(result.Values) != 384 || result.Values[0] != 1 {
			t.Fatalf("text: %+v %v", result, callErr)
		}
	}
	var encoded bytes.Buffer
	if err = png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 7, 11))); err != nil {
		t.Fatal(err)
	}
	imageResult, err := model.EncodeImageBytes(encoded.Bytes(), 0)
	if err != nil || len(imageResult.Values) != 384 {
		t.Fatalf("image: %+v %v", imageResult, err)
	}
	audioResult, err := model.EncodeAudioPCM(make([]float32, 100), 44100, 2, 0)
	if err != nil || len(audioResult.Values) != 384 {
		t.Fatalf("original audio: %+v %v", audioResult, err)
	}
	for _, dimension := range []int{32, 385} {
		if _, err = model.EncodeText("hello", dimension); err == nil {
			t.Fatalf("accepted unsupported dimension %d", dimension)
		}
	}
	if _, err = model.RuntimeDescriptor(1, 384); err == nil {
		t.Fatal("advertised nonexistent early exit")
	}
	if _, err = model.EncodeText(strings.Repeat("hello ", 30), 0); err == nil {
		t.Fatal("silently truncated overlength text")
	}
	for _, test := range []struct {
		pcm            []float32
		rate, channels int
	}{
		{[]float32{1}, 22050, 1}, {[]float32{float32(math.NaN())}, 16000, 1}, {[]float32{1}, 16000, 2}, {[]float32{1}, 0, 1},
	} {
		if _, err = model.EncodeAudioPCM(test.pcm, test.rate, test.channels, 0); err == nil {
			t.Fatal("accepted invalid PCM or unsupported original rate")
		}
	}
	if _, err = (&MultiModalModel{model.owner}).EncodeAudio(make([]float32, 80), 80, 1, 0); err == nil {
		t.Fatal("accepted legacy mel as original PCM")
	}
	if windows, windowErr := model.Windows("hello world", 0); windowErr != nil || len(windows) != 1 {
		t.Fatalf("windows: %+v %v", windows, windowErr)
	}
	// Identity is captured once. No manifest reread, even after its directory moves.
	moved := options.ModelPath + "-moved"
	if err = os.Rename(options.ModelPath, moved); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(moved)
	after, err := clone.RuntimeDescriptor(0, 384)
	if err != nil || after != raw {
		t.Fatalf("metadata was reloaded: %v", err)
	}
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = model.EncodeText("hello", 0); err == nil {
		t.Fatal("closed owner accepted inference")
	}
	if _, err = clone.EncodeText("hello", 0); err != nil {
		t.Fatal("closing original destroyed clone", err)
	}
}
func TestOmniIdentityIgnoresArtifactLocation(t *testing.T) {
	first, err := LoadOmni(omniFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := LoadOmni(omniFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	a, err := first.RuntimeDescriptor(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	b, err := second.RuntimeDescriptor(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Fatal("copied identical deployment received a different content identity")
	}
}
func TestOmniRejectsIncompleteAndWrongArtifacts(t *testing.T) {
	for name, change := range map[string]func(map[string]any){
		"format":           func(m map[string]any) { m["format_version"] = 2 },
		"pooling":          func(m map[string]any) { m["embedding"].(map[string]any)["text_pooling"] = "mean" },
		"unknown":          func(m map[string]any) { m["fallback"] = true },
		"missing modality": func(m map[string]any) { delete(m["graphs"].(map[string]any), "clap") },
		"no parity":        func(m map[string]any) { m["reference_parity"].(map[string]any)["passed"] = false },
		"wrong checksum":   func(m map[string]any) { m["files"].(map[string]any)["processors/audio.json"] = strings.Repeat("0", 64) },
		"unsafe tokenizer": func(m map[string]any) { m["tokenizer"] = "../tokenizer.json" },
	} {
		t.Run(name, func(t *testing.T) {
			options := omniFixture(t)
			changeOmniManifest(t, options, change)
			if model, err := LoadOmni(options); err == nil {
				model.Close()
				t.Fatal("accepted wrong contract")
			}
		})
	}
	t.Run("parity receipt source", func(t *testing.T) {
		options := omniFixture(t)
		path := filepath.Join(options.ModelPath, "reference_parity.json")
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var receipt map[string]any
		if err = json.Unmarshal(raw, &receipt); err != nil {
			t.Fatal(err)
		}
		receipt["source"].(map[string]any)["revision"] = strings.Repeat("f", 40)
		raw, err = json.Marshal(receipt)
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(path, raw, 0600); err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256(raw)
		changeOmniManifest(t, options, func(m map[string]any) {
			m["files"].(map[string]any)["reference_parity.json"] = hex.EncodeToString(hash[:])
		})
		if model, err := LoadOmni(options); err == nil {
			model.Close()
			t.Fatal("accepted parity for a different source revision")
		}
	})
	t.Run("actual graph ports", func(t *testing.T) {
		options := omniFixture(t)
		path := filepath.Join(options.ModelPath, "onnx/image.onnx")
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		raw = bytes.ReplaceAll(raw, []byte("embedding"), []byte("wrongname"))
		if err = os.WriteFile(path, raw, 0600); err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256(raw)
		changeOmniManifest(t, options, func(m map[string]any) { m["files"].(map[string]any)["onnx/image.onnx"] = hex.EncodeToString(hash[:]) })
		if model, err := LoadOmni(options); err == nil {
			model.Close()
			t.Fatal("accepted graph output inconsistent with manifest")
		}
	})
	t.Run("missing CLAP", func(t *testing.T) {
		options := omniFixture(t)
		if err := os.Remove(filepath.Join(options.ModelPath, "onnx/clap.onnx")); err != nil {
			t.Fatal(err)
		}
		if model, err := LoadOmni(options); err == nil {
			model.Close()
			t.Fatal("silently dropped CLAP")
		}
	})
	t.Run("truncation", func(t *testing.T) {
		options := omniFixture(t)
		options.Overflow = "truncate_right"
		if model, err := LoadOmni(options); err == nil {
			model.Close()
			t.Fatal("accepted a silent-truncation deployment")
		}
	})
}

func TestOmniExplicitFixedExecutionResolvesSessionShape(t *testing.T) {
	options := omniFixture(t)
	options.ExecutionMaxInputTokens = 32
	model, err := LoadOmni(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	if _, err := model.EncodeText("hello", 0); err != nil {
		t.Fatal(err)
	}
	info, err := model.Info()
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, session := range info.Sessions {
		for _, input := range session.InputSchema {
			if input.Name == "input_ids" {
				found = true
				if !reflect.DeepEqual(input.Shape, []int64{1, 32}) {
					t.Fatalf("fixed budget was not resolved before session preparation: %+v", input)
				}
				if session.ExecutionMaxInputTokens != 32 || len(session.ExecutionInputs) != len(session.InputSchema) {
					t.Fatalf("fixed execution contract missing from evidence: %+v", session)
				}
				for _, contract := range session.ExecutionInputs {
					if contract.Name == input.Name && !reflect.DeepEqual(contract, input) {
						t.Fatalf("execution contract differs from loaded session: %+v vs %+v", contract, input)
					}
				}
			}
		}
	}
	if !found {
		t.Fatal("missing text session input evidence")
	}
	if _, err := model.EncodeText(strings.Repeat("hello ", 40), 0); err == nil {
		t.Fatal("fixed session silently truncated input")
	}
}
