//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

type omniArray struct {
	File  string `json:"file"`
	Shape []int  `json:"shape"`
}

func readOmniArray(t *testing.T, root string, array omniArray) []float32 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(root, array.File))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw)%4 != 0 {
		t.Fatal("invalid golden float32 length")
	}
	values := make([]float32, len(raw)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return values
}

func compareOmniVector(t *testing.T, name string, actual, expected []float32) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("%s dimension mismatch", name)
	}
	var dot, aNorm, bNorm, worst float64
	for i, a := range actual {
		b := expected[i]
		delta := math.Abs(float64(a - b))
		worst = math.Max(worst, delta)
		if math.IsNaN(float64(a)) || math.IsInf(float64(a), 0) || delta > 1e-4+2e-4*math.Abs(float64(b)) {
			t.Fatalf("%s[%d]: %g vs %g", name, i, a, b)
		}
		dot += float64(a) * float64(b)
		aNorm += float64(a) * float64(a)
		bNorm += float64(b) * float64(b)
	}
	cosine := dot / math.Sqrt(aNorm*bNorm)
	if cosine < 0.99999 {
		t.Fatalf("%s cosine %g", name, cosine)
	}
	t.Logf("%s max error %.8g cosine %.9g", name, worst, cosine)
}

func publishedOmniOptions(t *testing.T) Options {
	t.Helper()
	root := os.Getenv("VELA_OMNI_ARTIFACT")
	if root == "" {
		t.Skip("set VELA_OMNI_ARTIFACT to a prepared artifact with reference goldens")
	}
	options := Options{ModelPath: root, Provider: "cpu", IntraThreads: 4}
	if provider := os.Getenv("VELA_OMNI_PROVIDER"); provider != "" {
		options.Provider = provider
	}
	if profile := os.Getenv("VELA_OMNI_PROFILE"); profile != "" {
		options.ProfilePrefix = profile
	}
	if budget := os.Getenv("VELA_OMNI_EXECUTION_TOKENS"); budget != "" {
		value, err := strconv.Atoi(budget)
		if err != nil || value < 1 {
			t.Fatal("invalid VELA_OMNI_EXECUTION_TOKENS")
		}
		options.ExecutionMaxInputTokens = value
	}
	if threads := os.Getenv("VELA_OMNI_THREADS"); threads != "" {
		value, err := strconv.Atoi(threads)
		if err != nil || value < 1 {
			t.Fatal("invalid VELA_OMNI_THREADS")
		}
		options.IntraThreads = value
	}
	if options.Provider != "cpu" && options.ProfilePrefix == "" {
		t.Fatal("GPU parity requires VELA_OMNI_PROFILE for actual node-placement evidence")
	}
	return options
}

// Explicit artifacts contain the export's source-reference goldens. No model
// downloads or heavyweight imports occur in tests.
func TestPublishedOmniParity(t *testing.T) {
	options := publishedOmniOptions(t)
	root := options.ModelPath
	model, err := LoadOmni(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	golden, index := readPublishedOmniGoldens(t, root)
	initialInfo, err := model.Info()
	if err != nil {
		t.Fatal(err)
	}
	completedText := 0
	for i, record := range index.Texts {
		text := record.Text
		if record.FormattedText != "" {
			text = record.FormattedText
		}
		out, err := model.EncodeText(text, 0)
		if record.Tokens > initialInfo.EffectiveLimit {
			var boundary *Error
			if !errors.As(err, &boundary) || boundary.Kind != "input_limit" {
				t.Fatalf("text%d with %d tokens must reject deployment budget %d with input_limit: %v", i, record.Tokens, initialInfo.EffectiveLimit, err)
			}
			t.Logf("text%d: %d tokens rejected by explicit %d-token deployment budget; not a numerical qualification", i, record.Tokens, initialInfo.EffectiveLimit)
			continue
		}
		if err != nil {
			t.Fatalf("text%d: %v", i, err)
		}
		completedText++
		compareOmniVector(t, "text", out.Values, readOmniArray(t, golden, record.Embedding))
	}
	for i, record := range index.Images {
		data, err := os.ReadFile(filepath.Join(golden, record.File))
		if err != nil {
			t.Fatal(err)
		}
		out, err := model.EncodeImageBytes(data, 0)
		if err != nil {
			t.Fatalf("image%d: %v", i, err)
		}
		compareOmniVector(t, "image", out.Values, readOmniArray(t, golden, record.Embedding))
	}
	for i, record := range index.Audio {
		channels := 1
		if len(record.PCM.Shape) == 2 {
			channels = record.PCM.Shape[0]
		}
		out, err := model.EncodeAudioPCM(readOmniArray(t, golden, record.PCM), record.SamplingRate, channels, 0)
		if err != nil {
			t.Fatalf("audio%d: %v", i, err)
		}
		compareOmniVector(t, "audio", out.Values, readOmniArray(t, golden, record.Embedding))
	}
	info, err := model.Info()
	if err != nil {
		t.Fatal(err)
	}
	if len(info.Sessions) != 4 || info.CompletedInferences != uint64(completedText+len(index.Images)+len(index.Audio)) {
		t.Fatalf("actual inference evidence incomplete: %+v", info)
	}
	auditPublishedOmniProfiles(t, model, options, info)
}

func auditPublishedOmniProfiles(t *testing.T, model *OmniModel, options Options, info Info) {
	t.Helper()
	paths, err := model.FinishProfiling()
	if err != nil {
		t.Fatal(err)
	}
	auditPublishedGPUProfiles(t, paths, options, info, 4)
}

func auditPublishedGPUProfiles(t *testing.T, paths []string, options Options, info Info, expected int) {
	t.Helper()
	if options.Provider != "cpu" {
		provider := map[string]string{"rocm": "ROCMExecutionProvider", "migraphx": "MIGraphXExecutionProvider"}[options.Provider]
		for _, session := range info.Sessions {
			if !session.CPUFallbackDisabled || session.Provider != provider {
				t.Fatalf("requested execution policy was not enforced: %+v", session)
			}
		}
		if len(paths) != expected || len(info.Sessions) != expected {
			t.Fatalf("missing per-graph profile: %v", paths)
		}
		if provider == "" {
			t.Fatalf("unsupported GPU parity provider %s", options.Provider)
		}
		for _, path := range paths {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var events []struct {
				Args struct {
					Provider string `json:"provider"`
				} `json:"args"`
			}
			if err := json.Unmarshal(raw, &events); err != nil {
				t.Fatal(err)
			}
			count := 0
			for _, event := range events {
				if event.Args.Provider == "CPUExecutionProvider" {
					t.Fatalf("CPU fallback node executed in %s", path)
				}
				if event.Args.Provider == provider {
					count++
				}
			}
			if count == 0 {
				t.Fatalf("no GPU nodes executed in %s", path)
			}
			t.Logf("%s: %d %s nodes; 0 CPU nodes", filepath.Base(path), count, provider)
		}
	}
}
