package classification

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// Exercise the same request-facts dispatch used by Preview and ExtProc, after
// ordinary candidate warmup. No native libraries or external models are needed.
func cancellationClassifier(t *testing.T, family string, provider embedding.Provider) *Classifier {
	t.Helper()
	c := &Classifier{Config: &config.RouterConfig{}}
	var err error
	if family == config.SignalTypeEmbedding {
		c.Config.EmbeddingRules = []config.EmbeddingRule{{Name: "topic", Candidates: []string{"anchor"}, SimilarityThreshold: .5}}
		c.keywordEmbeddingClassifier, err = NewEmbeddingClassifierWithProvider(c.Config.EmbeddingRules,
			config.HNSWConfig{ModelType: config.EmbeddingModelTypeRemote, PreloadEmbeddings: true}, provider)
		if err == nil {
			err = c.keywordEmbeddingClassifier.WarmupCandidateEmbeddings()
		}
	} else {
		c.Config.ComplexityRules = []config.ComplexityRule{{
			Name: "difficulty", Threshold: .1,
			Hard: config.ComplexityCandidates{Candidates: []string{"anchor"}},
			Easy: config.ComplexityCandidates{Candidates: []string{"easy"}},
		}}
		c.complexityClassifier, err = NewComplexityClassifier(c.Config.ComplexityRules,
			config.EmbeddingModelTypeRemote, config.PrototypeScoringConfig{}, provider)
	}
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func evaluateCancellationRequest(c *Classifier, ctx context.Context) *SignalResults {
	return c.EvaluateAllSignalsWithRequestFacts("request", "request", "request", nil, nil,
		false, true, "", nil, ConversationFacts{}, "", RequestFacts{Context: ctx})
}

func requireCanceledEmbeddingSignal(t *testing.T, result *SignalResults) {
	t.Helper()
	if len(result.SignalErrors) == 0 || len(result.SignalValues) != 0 ||
		len(result.MatchedEmbeddingRules)+len(result.MatchedComplexityRules) != 0 {
		t.Fatalf("canceled inference published evidence: %+v", result)
	}
}

type observedEmbeddingGate struct {
	admission.Admissioner
	entered chan struct{}
}

func (g observedEmbeddingGate) Acquire(ctx context.Context) (admission.Ticket, error) {
	g.entered <- struct{}{}
	return g.Admissioner.Acquire(ctx)
}

type cancellationModel struct{ closes atomic.Int32 }

func (m *cancellationModel) Close() error { m.closes.Add(1); return nil }

func TestEmbeddingSignalsCancelQueuedInference(t *testing.T) {
	for _, family := range []string{config.SignalTypeEmbedding, config.SignalTypeComplexity} {
		t.Run(family, func(t *testing.T) {
			gate := admission.NewSemaphore(1, 1, 0, admission.OverflowShed)
			entered := make(chan struct{}, 1)
			resource, err := binding.NewPool().Acquire(context.Background(), binding.ResourceIdentity{
				Artifact: "test-embedding", Revision: "one", Provider: "test", Device: "cpu", Precision: "fp32",
			}, "1/1/shed", observedEmbeddingGate{gate, entered}, func(context.Context) (io.Closer, error) {
				return &cancellationModel{}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			defer resource.Close()
			var forwards atomic.Int32
			provider, err := embedding.NewFuncProvider("test", 2, func(ctx context.Context, text string) ([]float32, error) {
				if text != "request" {
					return []float32{1, 0}, nil
				}
				useErr := resource.Use(ctx, func(io.Closer) error { forwards.Add(1); return nil })
				return []float32{1, 0}, useErr
			})
			if err != nil {
				t.Fatal(err)
			}
			c := cancellationClassifier(t, family, provider)
			hold, err := gate.Acquire(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			defer hold()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan *SignalResults, 1)
			go func() { done <- evaluateCancellationRequest(c, ctx) }()
			<-entered
			cancel()
			select {
			case result := <-done:
				requireCanceledEmbeddingSignal(t, result)
			case <-time.After(time.Second):
				hold()
				<-done
				t.Fatal("request cancellation did not reach provider admission")
			}
			if forwards.Load() != 0 {
				t.Fatal("canceled queued request entered native inference")
			}
		})
	}
}

func TestEmbeddingSignalsCancelProviderHTTP(t *testing.T) {
	for _, family := range []string{config.SignalTypeEmbedding, config.SignalTypeComplexity} {
		t.Run(family, func(t *testing.T) {
			started, canceled, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body struct{ Input []string }
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					return
				}
				if body.Input[0] == "request" {
					close(started)
					select {
					case <-r.Context().Done():
						close(canceled)
					case <-release:
					}
					return
				}
				_, _ = io.WriteString(w, `{"data":[{"index":0,"embedding":[1,0]}]}`)
			}))
			defer server.Close()
			defer close(release)
			provider, err := embedding.NewOpenAICompatibleProvider(embedding.OpenAICompatibleConfig{
				BaseURL: server.URL, Model: "embedding", TimeoutSeconds: 5,
			})
			if err != nil {
				t.Fatal(err)
			}
			c := cancellationClassifier(t, family, provider)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan *SignalResults, 1)
			go func() { done <- evaluateCancellationRequest(c, ctx) }()
			<-started
			cancel()
			select {
			case <-canceled:
			case <-time.After(time.Second):
				t.Fatal("request cancellation did not reach the embedding HTTP request")
			}
			requireCanceledEmbeddingSignal(t, <-done)
		})
	}
}

func TestEmbeddingSignalsDrainStartedNativeInference(t *testing.T) {
	for _, family := range []string{config.SignalTypeEmbedding, config.SignalTypeComplexity} {
		t.Run(family, func(t *testing.T) {
			model := &cancellationModel{}
			gate := admission.NewSemaphore(1, 1, 0, admission.OverflowShed)
			resource, err := binding.NewPool().Acquire(context.Background(), binding.ResourceIdentity{
				Artifact: "test-embedding", Revision: "one", Provider: "test", Device: "cpu", Precision: "fp32",
			}, "1/1/shed", gate, func(context.Context) (io.Closer, error) { return model, nil })
			if err != nil {
				t.Fatal(err)
			}
			defer resource.Close()
			started, finish := make(chan struct{}), make(chan struct{})
			drain := sync.OnceFunc(func() { close(finish) })
			defer drain()
			provider, err := embedding.NewFuncProvider("test", 2, func(ctx context.Context, text string) ([]float32, error) {
				if text != "request" {
					return []float32{1, 0}, nil
				}
				useErr := resource.Use(ctx, func(io.Closer) error { close(started); <-finish; return nil })
				return []float32{1, 0}, useErr
			})
			if err != nil {
				t.Fatal(err)
			}
			c := cancellationClassifier(t, family, provider)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done, closed := make(chan *SignalResults, 1), make(chan struct{})
			go func() { done <- evaluateCancellationRequest(c, ctx) }()
			<-started
			cancel()
			go func() { _ = resource.Close(); close(closed) }()
			waitCtx, stopWait := context.WithTimeout(context.Background(), 20*time.Millisecond)
			defer stopWait()
			if ticket, err := gate.Acquire(waitCtx); !errors.Is(err, context.DeadlineExceeded) {
				if ticket != nil {
					ticket()
				}
				t.Errorf("native admission released before drain: %v", err)
			}
			var result *SignalResults
			select {
			case result = <-done:
				t.Error("classification returned before native inference drained")
			default:
			}
			if model.closes.Load() != 0 {
				t.Error("model unloaded before native inference drained")
			}
			drain()
			if result == nil {
				result = <-done
			}
			requireCanceledEmbeddingSignal(t, result)
			<-closed
			if model.closes.Load() != 1 {
				t.Fatal("model was not released exactly once after native drain")
			}
		})
	}
}

type cancellableImageProvider struct {
	embedding.Provider
	image func(context.Context) ([]float32, error)
}

func (p cancellableImageProvider) EmbedImage(ctx context.Context, _ []byte, _ int) ([]float32, error) {
	return p.image(ctx)
}

func TestEmbeddingSignalsCancelImageProvider(t *testing.T) {
	for _, family := range []string{config.SignalTypeEmbedding, config.SignalTypeComplexity} {
		t.Run(family, func(t *testing.T) {
			started, release := make(chan struct{}), make(chan struct{})
			defer close(release)
			provider := cancellableImageProvider{Provider: &stubEmbeddingProvider{}, image: func(ctx context.Context) ([]float32, error) {
				close(started)
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case <-release:
					return []float32{1, 0}, nil
				}
			}}
			c := cancellationClassifier(t, family, provider)
			if family == config.SignalTypeEmbedding {
				c.Config.EmbeddingRules[0].QueryModality = config.QueryModalityImage
				c.keywordEmbeddingClassifier.rulesByModality = buildRulesByModality(c.Config.EmbeddingRules)
			} else {
				c.complexityClassifier.hasImageCandidates = true
				c.complexityClassifier.multiModalProvider = provider
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan *SignalResults, 1)
			go func() {
				done <- c.EvaluateAllSignalsWithRequestFacts("request", "request", "request", nil, nil,
					false, true, "", nil, ConversationFacts{}, "data:image/png;base64,aW1hZ2U=", RequestFacts{Context: ctx})
			}()
			<-started
			cancel()
			select {
			case result := <-done:
				requireCanceledEmbeddingSignal(t, result)
			case <-time.After(time.Second):
				t.Fatal("request cancellation did not reach the image provider")
			}
		})
	}
}

func TestEmbeddingCanceledLazyWarmupCanRetry(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var calls atomic.Int32
	provider, err := embedding.NewFuncProvider("test", 2, func(ctx context.Context, text string) ([]float32, error) {
		if text == "anchor" && calls.Add(1) == 1 {
			cancel()
			return nil, ctx.Err()
		}
		return []float32{1, 0}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	c, err := NewEmbeddingClassifierWithProvider([]config.EmbeddingRule{{Name: "topic", Candidates: []string{"anchor"}, SimilarityThreshold: .5}},
		config.HNSWConfig{ModelType: config.EmbeddingModelTypeRemote}, provider)
	if err != nil {
		t.Fatal(err)
	}
	if _, classifyErr := c.ClassifyDetailedWithContext(ctx, "request"); !errors.Is(classifyErr, context.Canceled) {
		t.Fatalf("lazy warmup lost cancellation: %v", classifyErr)
	}
	if c.preloadComplete || len(c.candidateEmbeddings) != 0 {
		t.Fatal("canceled warmup published partial candidate state")
	}
	result, err := c.ClassifyDetailed("request")
	if err != nil || len(result.Matches) != 1 {
		t.Fatalf("fresh request could not retry candidate preparation: result=%+v err=%v", result, err)
	}
}

func TestEmbeddingMixedModalitiesDiscardPartialResultOnCancellation(t *testing.T) {
	for _, cancelRequest := range []bool{true, false} {
		name := "ordinary_image_error_preserves_text"
		if cancelRequest {
			name = "image_cancellation_discards_text"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var imageCalls int
			provider := cancellableImageProvider{
				Provider: &stubEmbeddingProvider{embeddings: map[string][]float32{
					"request": {1, 0}, "anchor": {1, 0}, "image anchor": {1, 0},
				}},
				image: func(ctx context.Context) ([]float32, error) {
					imageCalls++
					if cancelRequest {
						cancel()
						return nil, ctx.Err()
					}
					return nil, errors.New("image encoder failed")
				},
			}
			cfg := &config.RouterConfig{}
			cfg.EmbeddingRules = []config.EmbeddingRule{
				{Name: "topic", Candidates: []string{"anchor"}, SimilarityThreshold: .5},
				{Name: "image", Candidates: []string{"image anchor"}, SimilarityThreshold: .5, QueryModality: config.QueryModalityImage},
			}
			ec, err := NewEmbeddingClassifierWithProvider(cfg.EmbeddingRules,
				config.HNSWConfig{ModelType: config.EmbeddingModelTypeRemote, PreloadEmbeddings: true}, provider)
			if err != nil {
				t.Fatal(err)
			}
			if warmupErr := ec.WarmupCandidateEmbeddings(); warmupErr != nil {
				t.Fatal(warmupErr)
			}
			c := &Classifier{Config: cfg, keywordEmbeddingClassifier: ec}
			result := c.EvaluateAllSignalsWithRequestFacts("request", "request", "request", nil, nil,
				false, true, "", nil, ConversationFacts{}, "data:image/png;base64,aW1hZ2U=", RequestFacts{Context: ctx})
			if imageCalls != 1 || result.SignalErrors["embedding:image"] != embeddingEvaluationFailedCode {
				t.Fatalf("image result: calls=%d errors=%v", imageCalls, result.SignalErrors)
			}
			if cancelRequest {
				requireCanceledEmbeddingSignal(t, result)
				if result.SignalErrors["embedding:topic"] != embeddingEvaluationFailedCode {
					t.Fatal("canceled text result was not marked as failed")
				}
			} else if len(result.MatchedEmbeddingRules) != 1 || result.MatchedEmbeddingRules[0] != "topic" ||
				result.SignalValues["embedding:topic"] != 1 || len(result.SignalErrors) != 1 {
				t.Fatalf("ordinary image failure discarded successful text: %+v", result)
			}
		})
	}
}
