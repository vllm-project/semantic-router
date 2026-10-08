package classification

import (
	"context"
	"io"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type nopCloser struct{}

func (nopCloser) Close() error { return nil }

// TestHallucinationDetectorSpansPointAtTheirOwnChunk checks that a span found in
// a later chunk of a long answer is reported where that chunk sits, even when
// the same text also appears earlier in the answer, as it does when a model
// repeats itself.
func TestHallucinationDetectorSpansPointAtTheirOwnChunk(t *testing.T) {
	ctx := context.Background()
	task, err := binding.Register(binding.NewRegistry(), config.RemoteClassifierContractTokenSpans,
		func(tasks.GroundedTextRequest) error { return nil },
		func(tasks.GroundedTextRequest, tasks.TokenClassificationResult) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	resource, err := binding.NewPool().Acquire(ctx, binding.ResourceIdentity{Artifact: "halu", Provider: "test", Device: "cpu", Precision: "fp32"}, "one", nil,
		func(context.Context) (io.Closer, error) { return nopCloser{}, nil })
	if err != nil {
		t.Fatal(err)
	}
	// The fake model flags each chunk's last sentence, so every chunk yields
	// exactly one span at a known chunk-relative offset.
	const sentence = "The tower is 330 metres tall."
	handle, err := task.Resolve(binding.Identity{Recipe: "default", Name: "hallucination_detector", Deployment: "fixture", Contract: config.RemoteClassifierContractTokenSpans, Adapter: "modernbert"},
		binding.Capability{Contract: config.RemoteClassifierContractTokenSpans, Provider: "test", Device: "cpu", Precision: "fp32"}, resource,
		func(_ context.Context, _ io.Closer, input tasks.GroundedTextRequest) (tasks.TokenClassificationResult, error) {
			start := strings.LastIndex(input.Answer, sentence)
			return tasks.TokenClassificationResult{Entities: []tasks.TokenEntity{{EntityType: "hallucinated", Start: start, End: start + len(sentence), Text: sentence}}}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	detector := &HallucinationDetector{
		config:      &config.HallucinationModelConfig{},
		spec:        config.ResolvedModelBinding{Recipe: "default", Binding: config.ModelBinding{Adapter: "modernbert"}},
		handle:      handle,
		initialized: true,
	}

	answer := strings.Repeat("It stands in Paris. "+sentence+" ", 120)
	chunks := hallucinationAnswerChunks(answer)
	if len(chunks) < 3 {
		t.Fatalf("fixture needs several chunks, got %d", len(chunks))
	}
	result, err := detector.Detect(ctx, "The tower is 330 metres tall.", "How tall is the tower?", answer)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Spans) != len(chunks) {
		t.Fatalf("got %d spans for %d chunks: chunks mapped onto the same position", len(result.Spans), len(chunks))
	}
	for i, span := range result.Spans {
		want := securitySignalChunkSpans(answer, hallucinationAnswerChunkBudget, hallucinationAnswerOverlapRunes)[i]
		wantStart := want.StartByte + strings.LastIndex(want.Text, sentence)
		if span.Start != wantStart || answer[span.Start:span.End] != span.Text {
			t.Fatalf("span %d at %d, want %d (end of chunk %d)", i, span.Start, wantStart, i)
		}
	}
}
