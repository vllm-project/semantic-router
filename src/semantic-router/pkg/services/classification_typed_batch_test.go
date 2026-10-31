package services

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type typedBatchImageProvider struct {
	*embedding.FuncProvider
	image      func(context.Context, []byte) ([]float32, error)
	calls      atomic.Int32
	modalities []string
}

func (p *typedBatchImageProvider) EmbedImage(ctx context.Context, data []byte, _ int) ([]float32, error) {
	p.calls.Add(1)
	if p.image != nil {
		return p.image(ctx, data)
	}
	return []float32{1, 0}, nil
}

func (p *typedBatchImageProvider) EmbeddingInfo() embedding.ModelInfo {
	return embedding.ModelInfo{Dimension: 2, Dimensions: []int{2}, Modalities: p.modalities}
}

func typedBatchService(t *testing.T, provider *typedBatchImageProvider, configure ...func(*config.RouterConfig)) *ClassificationService {
	t.Helper()
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.EmbeddingConfig.TargetDimension = 2
	cfg.EmbeddingRules = []config.EmbeddingRule{
		{Name: "image_rule", Candidates: []string{"anchor"}, SimilarityThreshold: 0.5, QueryModality: config.QueryModalityImage},
		{Name: "text_rule", Candidates: []string{"anchor"}, SimilarityThreshold: 0.5},
	}
	cfg.Decisions = []config.Decision{
		{Name: "image_route", Rules: config.RuleNode{Type: "embedding", Name: "image_rule"}},
		{Name: "text_route", Rules: config.RuleNode{Type: "embedding", Name: "text_rule"}},
	}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"vision": {Provider: config.ModelRuntimeProvider, Device: "cpu", Artifact: t.TempDir()}}
	cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "vision", Contract: "embedding.v1", Adapter: "vela_omni"}}
	for _, apply := range configure {
		apply(cfg)
	}
	prepared := embedding.NewSet(map[string]embedding.Provider{"mmbert": provider}, "mmbert")
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil, classification.RecipeRuntimeOptions{Embeddings: prepared})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifiers.Close() })
	return NewRecipeClassificationService(classifiers, cfg)
}

func typedBatchProvider(t *testing.T) *typedBatchImageProvider {
	t.Helper()
	fp, err := embedding.NewFuncProvider("synthetic", 2, func(ctx context.Context, text string) ([]float32, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return []float32{1, 0}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return &typedBatchImageProvider{FuncProvider: fp, modalities: []string{"text", "image"}}
}

func typedInputs(t *testing.T, body string) []ClassificationInput {
	t.Helper()
	var inputs []ClassificationInput
	if err := json.Unmarshal([]byte(body), &inputs); err != nil {
		t.Fatal(err)
	}
	return inputs
}

func TestTypedBatchImageFailureKeepsOrderedSiblings(t *testing.T) {
	for _, failed := range []int{0, 1, 2} {
		t.Run(strconv.Itoa(failed), func(t *testing.T) {
			provider := typedBatchProvider(t)
			provider.image = func(ctx context.Context, data []byte) ([]float32, error) {
				index := int(data[0] - 'a')
				select {
				case <-time.After(time.Duration(3-index) * 5 * time.Millisecond):
				case <-ctx.Done():
					return nil, ctx.Err()
				}
				if index == failed {
					return nil, errors.New("image decoder failed")
				}
				return []float32{1, 0}, nil
			}
			service := typedBatchService(t, provider)
			result, err := service.ClassifyInputsContext(context.Background(), typedInputs(t, `[{"type":"image","image_url":"data:image/png;base64,YQ=="},{"type":"image","image_url":"data:image/png;base64,Yg=="},{"type":"image","image_url":"data:image/png;base64,Yw=="}]`), "intent", nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Results) != 3 {
				t.Fatalf("lost items: %+v", result)
			}
			for index, item := range result.Results {
				if index == failed {
					if item.Error == nil || item.Error.Code != "CLASSIFICATION_ERROR" {
						t.Fatalf("failure %d: %+v", index, item)
					}
					continue
				}
				if item.Error != nil || item.Intent == nil || item.Intent.Classification.Category != "image_route" {
					t.Fatalf("sibling %d: %+v", index, item)
				}
			}
		})
	}
}

func TestTypedBatchMixedInputsAndPreflight(t *testing.T) {
	provider := typedBatchProvider(t)
	service := typedBatchService(t, provider)
	inputs := typedInputs(t, `[{"type":"text","text":"hello"},{"type":"image","image_url":"data:image/png;base64,YQ=="},{"type":"image","image_url":"https://example.com/no-fetch"},null,{"type":"audio","text":"no"}]`)
	result, err := service.ClassifyInputsContext(context.Background(), inputs, "intent", nil)
	if err != nil {
		t.Fatal(err)
	}
	for index, want := range []string{"text_route", "image_route"} {
		item := result.Results[index]
		if item.Error != nil || item.Intent == nil || item.Intent.Classification.Category != want {
			t.Fatalf("item %d: %+v", index, item)
		}
	}
	for index := 2; index < len(inputs); index++ {
		if result.Results[index].Error == nil || result.Results[index].Error.Code != "INVALID_INPUT" {
			t.Fatalf("item %d: %+v", index, result.Results[index])
		}
	}
	if provider.calls.Load() != 1 {
		t.Fatalf("invalid images reached model: %d", provider.calls.Load())
	}
	for _, task := range []string{"pii", "security", "all"} {
		result, err = service.ClassifyInputsContext(context.Background(), inputs[1:2], task, nil)
		if err != nil || result.Results[0].Error == nil || result.Results[0].Error.Code != "UNSUPPORTED_INPUT" {
			t.Fatalf("%s: %+v %v", task, result, err)
		}
	}
	if provider.calls.Load() != 1 {
		t.Fatal("unsupported tasks performed image inference")
	}
	provider.modalities = []string{"text"}
	result, err = service.ClassifyInputsContext(context.Background(), inputs[1:2], "intent", nil)
	if err != nil || result.Results[0].Error == nil || result.Results[0].Error.Code != "UNSUPPORTED_INPUT" || provider.calls.Load() != 1 {
		t.Fatalf("text-only model accepted image: %+v %v", result, err)
	}
}

func TestTypedBatchCancellationAndRuntimeReplacement(t *testing.T) {
	for _, deadline := range []bool{false, true} {
		t.Run(map[bool]string{false: "cancel", true: "deadline"}[deadline], func(t *testing.T) {
			provider := typedBatchProvider(t)
			entered := make(chan struct{}, 1)
			provider.image = func(ctx context.Context, _ []byte) ([]float32, error) {
				entered <- struct{}{}
				<-ctx.Done()
				return nil, ctx.Err()
			}
			service := typedBatchService(t, provider)
			ctx, cancel := context.WithCancel(context.Background())
			if deadline {
				ctx, cancel = context.WithTimeout(context.Background(), 500*time.Millisecond)
			}
			defer cancel()
			done := make(chan *ClassificationInputsResponse, 1)
			inputs := typedInputs(t, `[{"type":"image","image_url":"data:image/png;base64,YQ=="}]`)
			go func() {
				result, err := service.ClassifyInputsContext(ctx, inputs, "intent", nil)
				if err != nil {
					t.Error(err)
				}
				done <- result
			}()
			select {
			case <-entered:
			case <-time.After(3 * time.Second):
				t.Fatal("image inference did not start")
			}
			replaced := make(chan error, 1)
			go func() { replaced <- service.TryRefreshRuntimeConfig(&config.RouterConfig{}) }()
			select {
			case err := <-replaced:
				t.Fatalf("replacement retired in-flight batch: %v", err)
			case <-time.After(20 * time.Millisecond):
			}
			if !deadline {
				cancel()
			}
			select {
			case result := <-done:
				want := "CANCELED"
				if deadline {
					want = "DEADLINE_EXCEEDED"
				}
				if result == nil || result.Results[0].Error == nil || result.Results[0].Error.Code != want {
					t.Fatalf("got %+v, want %s", result, want)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("cancellation did not drain batch")
			}
			select {
			case err := <-replaced:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("replacement did not finish after drain")
			}
		})
	}
}

func TestTypedBatchReportsCategoryFallbackFailure(t *testing.T) {
	var failing atomic.Bool
	service := domainResultService(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if failing.Load() {
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
			return
		}
		writeDomainDistribution(w)
	}), func(cfg *config.RouterConfig) { cfg.Decisions = nil })
	failing.Store(true)
	legacy, err := service.ClassifyIntent(context.Background(), IntentRequest{Text: "fallback"})
	if err != nil || legacy.Classification.Category != "other" {
		t.Fatalf("legacy fallback changed: %+v %v", legacy, err)
	}
	result, err := service.ClassifyInputsContext(context.Background(), typedInputs(t, `[{"type":"text","text":"fallback"}]`), "intent", nil)
	if err != nil || result.Results[0].Error == nil || result.Results[0].Intent != nil {
		t.Fatalf("failed fallback reported success: %+v %v", result, err)
	}
}

func TestTypedBatchCategoryFallbackPreservesCancellation(t *testing.T) {
	entered, escape := make(chan struct{}), make(chan struct{})
	defer close(escape)
	provider := typedBatchProvider(t)
	fp, err := embedding.NewFuncProvider("cancel-fallback", 2, func(ctx context.Context, text string) ([]float32, error) {
		if text != "wait" {
			return []float32{1, 0}, nil
		}
		close(entered)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-escape:
			return nil, errors.New("test cleanup")
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	provider.FuncProvider = fp
	service := typedBatchService(t, provider, func(cfg *config.RouterConfig) { cfg.Decisions = nil })
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	inputs := typedInputs(t, `[{"type":"text","text":"wait"}]`)
	done := make(chan *ClassificationInputsResponse, 1)
	go func() {
		result, err := service.ClassifyInputsContext(ctx, inputs, "intent", nil)
		if err != nil {
			t.Error(err)
		}
		done <- result
	}()
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("category fallback did not start")
	}
	cancel()
	select {
	case result := <-done:
		if result == nil || result.Results[0].Error == nil || result.Results[0].Error.Code != "CANCELED" {
			t.Fatalf("fallback did not retain cancellation: %+v", result)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("category fallback lost caller cancellation")
	}
}
