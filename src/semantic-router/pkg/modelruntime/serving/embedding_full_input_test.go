package serving

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// fullInputServices models the worker's tokenizer-before-forward contract.
// Truncating requests still run a forward; rejecting oversized requests do not.
type fullInputServices struct {
	*embedServices
	requests []modelservice.EmbedRequest
	forwards int
	coverage string
	failure  error
}

func (f *fullInputServices) Embed(ctx context.Context, deployment string, request modelservice.EmbedRequest) (modelservice.EmbedResponse, error) {
	f.requests = append(f.requests, request)
	if f.failure != nil {
		return modelservice.EmbedResponse{}, f.failure
	}
	for _, input := range request.Inputs {
		if request.Overflow == "reject" && len(strings.Fields(input.Text)) > request.MaxTokens {
			return modelservice.EmbedResponse{Embeddings: make([][]float32, len(request.Inputs)), Errors: []string{"max_length_exceeded"}}, nil
		}
	}
	f.forwards++
	response, err := f.embedServices.Embed(ctx, deployment, request)
	for i := range response.Inputs {
		switch f.coverage {
		case "missing":
			response.Inputs[i] = nil
		case "truncated":
			response.Inputs[i].Truncated = true
		case "inconsistent":
			response.Inputs[i].ProcessedTokens--
		case "unknown":
			response.Inputs[i] = &modelservice.InputUsage{}
		case "lower_bound":
			lowerBound := true
			response.Inputs[i].TokensLowerBound = &lowerBound
		}
	}
	return response, err
}

func prepareFullInputEmbedding(t *testing.T) (*EmbeddingProvider, *fullInputServices) {
	t.Helper()
	card := embeddingCard(t.Name(), "text")
	card.ModelSHA256 = fmt.Sprintf("%x", sha256.Sum256([]byte(t.Name())))
	services := &fullInputServices{embedServices: &embedServices{cards: map[string]modelservice.ModelCard{card.ID: card}}}
	spec := embeddingSpec(card.ID)
	spec.Deployment.Input.MaxTokens = 8
	provider, err := New(services, nil).Embedding(context.Background(), spec, 4, 6)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = provider.Close() })
	services.requests = nil
	services.forwards = 0
	return provider, services
}

func TestFullInputEmbeddingRejectsBeforeForwardAndSeparatesTruncatedCache(t *testing.T) {
	provider, services := prepareFullInputEmbedding(t)
	ctx := context.Background()
	long := strings.Repeat("oversized input ", 8)
	if vector, err := provider.Embed(ctx, long); err != nil || len(vector) != 4 {
		t.Fatalf("ordinary truncating input: %v %v", vector, err)
	}
	if services.forwards != 1 {
		t.Fatalf("ordinary forwards=%d", services.forwards)
	}
	for range 2 {
		if vector, err := provider.EmbedFullInput(ctx, long); vector != nil || !errors.Is(err, binding.ErrInputLimit) {
			t.Fatalf("oversized complete input: %v %v", vector, err)
		}
	}
	if services.forwards != 1 || len(services.requests) != 3 {
		t.Fatalf("strict rejection forwarded or reused truncated cache: forwards=%d requests=%d", services.forwards, len(services.requests))
	}
	for _, request := range services.requests[1:] {
		if request.Overflow != "reject" || request.MaxTokens != 8 || request.Dimensions != 4 || request.Layer != 6 || request.Inputs[0].Text != long {
			t.Fatalf("strict request changed input/view/budget: %+v", request)
		}
	}
	if _, err := provider.Embed(ctx, long); err != nil || len(services.requests) != 3 {
		t.Fatalf("ordinary cache or truncate contract changed: %v", err)
	}
}

func TestFullInputEmbeddingRetainsViewsAndCachesCompleteVectors(t *testing.T) {
	provider, services := prepareFullInputEmbedding(t)
	ctx := context.Background()
	view := embedding.WithOptions(provider, embedding.Options{Dimension: 4, Layer: 6})
	for range 2 {
		vector, err := embedding.EmbedFullInput(ctx, view, "complete view query", embedding.Options{})
		if err != nil || len(vector) != 4 || vector[2] != 6 {
			t.Fatalf("wrapped view: %v %v", vector, err)
		}
	}
	if services.forwards != 1 || len(services.requests) != 1 || services.requests[0].Overflow != "reject" {
		t.Fatalf("strict view not cached: %+v", services.requests)
	}
	vector, err := embedding.EmbedFullInput(ctx, view, "complete view query", embedding.Options{Dimension: 8, Layer: 22})
	if err != nil || len(vector) != 8 || vector[2] != 22 || services.forwards != 2 {
		t.Fatalf("explicit view ignored or reused incompatible vector: %v %v forwards=%d", vector, err, services.forwards)
	}
	vector, err = embedding.EmbedFullInput(ctx, provider, "complete view query", embedding.Options{})
	if err != nil || len(vector) != 8 || vector[2] != 0 {
		t.Fatalf("zero direct options no longer select full output: %v %v", vector, err)
	}
	nested := embedding.WithOptions(view, embedding.Options{})
	ordinary, ordinaryErr := nested.Embed(ctx, "nested full view")
	complete, completeErr := embedding.EmbedFullInput(ctx, nested, "nested full view", embedding.Options{})
	if ordinaryErr != nil || completeErr != nil || len(ordinary) != 8 || len(complete) != 8 || ordinary[2] != 0 || complete[2] != 0 {
		t.Fatalf("nested zero options differ: ordinary=%v/%v complete=%v/%v", ordinary, ordinaryErr, complete, completeErr)
	}
}

func TestFullInputEmbeddingRejectsUnprovenCoverageWithoutCaching(t *testing.T) {
	for _, coverage := range []string{"missing", "truncated", "inconsistent", "unknown", "lower_bound"} {
		t.Run(coverage, func(t *testing.T) {
			provider, services := prepareFullInputEmbedding(t)
			services.coverage = coverage
			ctx := context.Background()
			// Ordinary embeddings may cache a vector with incomplete metadata.
			if _, err := provider.Embed(ctx, "coverage query"); err != nil {
				t.Fatal(err)
			}
			for range 2 {
				if vector, err := provider.EmbedFullInput(ctx, "coverage query"); err == nil || vector != nil {
					t.Fatalf("unproven coverage accepted: %v %v", vector, err)
				}
			}
			if services.forwards != 3 {
				t.Fatalf("invalid coverage entered strict cache: forwards=%d", services.forwards)
			}
			services.coverage = ""
			if _, err := provider.EmbedFullInput(ctx, "coverage query"); err != nil || services.forwards != 4 {
				t.Fatalf("failed coverage poisoned recovery: %v forwards=%d", err, services.forwards)
			}
		})
	}
}

func TestFullInputEmbeddingPropagatesCancellationAndErrors(t *testing.T) {
	provider, services := prepareFullInputEmbedding(t)
	ctx := context.Background()
	if _, err := provider.EmbedFullInput(ctx, "cached query"); err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	for _, text := range []string{"cached query", "uncached query"} {
		if _, err := provider.EmbedFullInput(canceled, text); !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled complete input: %v", err)
		}
	}
	if len(services.requests) != 1 {
		t.Fatal("canceled request reached worker")
	}
	services.failure = context.DeadlineExceeded
	if _, err := provider.EmbedFullInput(ctx, "failed query"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("worker error lost: %v", err)
	}
	services.failure = nil
	if _, err := provider.EmbedFullInput(ctx, "failed query"); err != nil {
		t.Fatalf("failed call poisoned cache: %v", err)
	}
	_ = provider.Close()
	if _, err := provider.EmbedFullInput(ctx, "cached query"); !errors.Is(err, binding.ErrClosed) {
		t.Fatalf("closed provider served strict cached result: %v", err)
	}
}
