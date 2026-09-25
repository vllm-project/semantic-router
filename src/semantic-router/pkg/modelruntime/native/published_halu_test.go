package native

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type haluReferenceSpan struct {
	Text       string
	Start, End int
	Confidence float32
}
type haluReferenceProbe struct {
	Name, Context, Question, Answer string
	InputTokens                     int `json:"input_tokens"`
	Spans                           []haluReferenceSpan
}

// TestPublishedVelaHalu is mandatory for the supported Candle CPU inventory.
// Explicit ORT/reference qualification can reuse it with prepared artifacts;
// this test never downloads a model.
func TestPublishedVelaHalu(t *testing.T) {
	path := os.Getenv("VLLM_SR_HALU_MODEL")
	if path == "" {
		if os.Getenv("VLLM_SR_REQUIRE_HALU_TESTS") == "1" {
			t.Fatal("required Halu model path is absent")
		}
		t.Skip("published Halu requires VLLM_SR_HALU_MODEL")
	}
	path, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	provider := os.Getenv("VLLM_SR_MODEL_TEST_PROVIDER")
	if provider == "" {
		provider = "candle"
	}
	device := os.Getenv("VLLM_SR_MODEL_TEST_DEVICE")
	if device == "" {
		device = "cpu"
	}
	spec := config.ResolvedModelBinding{
		Recipe: "published-halu", Name: "hallucination_detector",
		Binding:    config.ModelBinding{Deployment: "halu", Adapter: "vela_halu", Contract: config.RemoteClassifierContractTokenSpans, Head: os.Getenv("VLLM_SR_HALU_HEAD")},
		Deployment: config.ModelDeployment{Artifact: path, Provider: provider, Device: device, Precision: "native", Input: config.ModelInputBudget{MaxTokens: 8192, Overflow: "reject"}},
	}
	model, err := New(nil).Grounded(context.Background(), spec, .5)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	probes := []haluReferenceProbe{
		{Name: "supported", Context: "The museum opens at 10:00 on Tuesday.", Question: "When does the museum open on Tuesday?", Answer: "The museum opens at 10:00 on Tuesday."},
		{Name: "unsupported", Context: "The museum opens at 10:00 on Tuesday.", Question: "When does the museum open on Tuesday?", Answer: "The museum opens at 09:00 on Tuesday."},
		{Name: "unicode", Context: "Élodie lives in Paris. 王明住在北京。", Question: "Where do Élodie and 王明 live?", Answer: "Élodie lives in Berlin. 王明住在上海。"},
	}
	reference := os.Getenv("VLLM_SR_HALU_REFERENCE")
	if reference != "" {
		data, readErr := os.ReadFile(reference)
		if readErr != nil {
			t.Fatal(readErr)
		}
		var fixture struct {
			Schema     string
			OffsetUnit string `json:"offset_unit"`
			Source     struct{ Revision string }
			Probes     []haluReferenceProbe
		}
		if err = json.Unmarshal(data, &fixture); err != nil {
			t.Fatal(err)
		}
		registered := config.GetModelByPath("models/Vela-1.0-Encoder-307M-Halu")
		if fixture.Schema != "vela_halu_reference.v1" || fixture.OffsetUnit != "utf8_bytes" || fixture.Source.Revision != registered.Revision || len(fixture.Probes) < 3 {
			t.Fatal("reference does not describe the pinned Halu contract")
		}
		probes = fixture.Probes
	}
	for index, probe := range probes {
		t.Run(probe.Name, func(t *testing.T) {
			result, callErr := model.Call(context.Background(), string(spec.Recipe), tasks.GroundedTextRequest{Context: probe.Context, Question: probe.Question, Answer: probe.Answer})
			if callErr != nil {
				t.Fatal(callErr)
			}
			if result.Input == nil || result.Input.Truncated || result.Input.ProcessedTokens > 8192 {
				t.Fatalf("invalid pair usage: %+v", result.Input)
			}
			if reference == "" {
				if (index == 0 && len(result.Entities) != 0) || (index > 0 && len(result.Entities) == 0) {
					t.Fatalf("unexpected semantic verdict: %+v", result.Entities)
				}
			} else {
				if result.Input.OriginalTokens != probe.InputTokens || len(result.Entities) != len(probe.Spans) {
					t.Fatalf("pair reference differs: usage=%+v entities=%+v want=%+v", result.Input, result.Entities, probe)
				}
				for i, actual := range result.Entities {
					want := probe.Spans[i]
					if actual.Start != want.Start || actual.End != want.End || actual.Text != want.Text || math.Abs(float64(actual.Confidence-want.Confidence)) > .01 {
						t.Errorf("span %d=%+v want=%+v", i, actual, want)
					}
				}
			}
			for _, span := range result.Entities {
				if span.Start < 0 || span.End > len(probe.Answer) || span.Start >= span.End ||
					!utf8.ValidString(span.Text) || probe.Answer[span.Start:span.End] != span.Text ||
					span.Confidence <= .5 || math.IsNaN(float64(span.Confidence)) || math.IsInf(float64(span.Confidence), 0) {
					t.Fatalf("invalid answer-relative UTF-8 hallucination span: %+v", span)
				}
			}
			t.Logf("provider=%s device=%s input=%+v spans=%+v", provider, device, result.Input, result.Entities)
		})
	}
	t.Run("input-budget", func(t *testing.T) {
		_, err = model.Call(context.Background(), string(spec.Recipe), tasks.GroundedTextRequest{Context: strings.Repeat("hello ", 9000), Answer: "The museum opens at 10:00."})
		if !errors.Is(err, binding.ErrInputLimit) {
			t.Fatalf("over-budget pair must reject before inference: %v", err)
		}
	})
}
