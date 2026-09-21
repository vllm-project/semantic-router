//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Explicit source-reference fixture; never downloads models or relaxes the
// execution provider policy. The native factory has its own public-path test.
func TestPublishedGroundedParity(t *testing.T) {
	root := os.Getenv("VELA_HALU_ARTIFACT")
	if root == "" {
		t.Skip("set VELA_HALU_ARTIFACT to the exported Halu artifact")
	}
	options := Options{ModelPath: root, Provider: "cpu", MaxInputTokens: 8192, IntraThreads: 4}
	if provider := os.Getenv("VELA_HALU_PROVIDER"); provider != "" {
		options.Provider = provider
	}
	options.ProfilePrefix = os.Getenv("VELA_HALU_PROFILE")
	if options.Provider != "cpu" {
		if options.ProfilePrefix == "" {
			t.Fatal("GPU qualification requires a node-placement profile")
		}
		options.ExecutionMaxInputTokens = options.MaxInputTokens
	}
	data, err := os.ReadFile(filepath.Join(root, "halu_reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var reference struct {
		Schema     string
		OffsetUnit string `json:"offset_unit"`
		Probes     []struct {
			Name, Context, Question, Answer string
			InputTokens                     int `json:"input_tokens"`
			Spans                           []struct {
				Text       string
				Start, End int
				Confidence float32
			}
		}
	}
	if err = json.Unmarshal(data, &reference); err != nil {
		t.Fatal(err)
	}
	if reference.Schema != "vela_halu_reference.v1" || reference.OffsetUnit != "utf8_bytes" || len(reference.Probes) < 3 {
		t.Fatal("invalid source reference contract")
	}
	model, err := LoadGroundedClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	for _, probe := range reference.Probes {
		t.Run(probe.Name, func(t *testing.T) {
			out, detectErr := model.Detect(probe.Context, probe.Question, probe.Answer)
			if detectErr != nil {
				t.Fatal(detectErr)
			}
			if out.Input.Truncated || out.Input.OriginalTokens != probe.InputTokens || out.Input.ProcessedTokens != probe.InputTokens || len(out.Spans) != len(probe.Spans) {
				t.Fatalf("input or spans differ: %+v", out)
			}
			for i, actual := range out.Spans {
				expected := probe.Spans[i]
				if actual.Text != expected.Text || actual.Start != expected.Start || actual.End != expected.End || math.Abs(float64(actual.Confidence-expected.Confidence)) > .01 {
					t.Fatalf("span %d differs: %+v vs %+v", i, actual, expected)
				}
			}
			t.Logf("%d original tokens, %d source-matching spans", probe.InputTokens, len(out.Spans))
		})
	}
	info, err := model.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.CompletedInferences != uint64(len(reference.Probes)) {
		t.Fatalf("missing completed execution evidence: %+v", info)
	}
	profiles, err := model.FinishProfiling()
	if err != nil {
		t.Fatal(err)
	}
	auditPublishedGPUProfiles(t, profiles, options, info, 1)
}
