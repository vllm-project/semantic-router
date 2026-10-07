package serving

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// RelevanceScorer serves a RAG reranker binding through a model_runtime
// deployment at one fixed pair-scorer exit. Scores are raw relevance logits
// in input order.
type RelevanceScorer struct {
	call      *binding.Resolved[[]tasks.QueryDocument, tasks.RelevanceScores]
	selection config.PairScorerSelection
	identity  string
}

// Close releases the binding's reference to its deployment.
func (s *RelevanceScorer) Close() error { return s.call.Close() }

// CacheIdentity names the scorer's model content and exit for ranking caches.
func (s *RelevanceScorer) CacheIdentity() string { return s.identity }

// Selection is the pair-scorer exit the binding runs.
func (s *RelevanceScorer) Selection() config.PairScorerSelection { return s.selection }

// ScorePairs scores query/document pairs.
func (s *RelevanceScorer) ScorePairs(ctx context.Context, recipe string, pairs []tasks.QueryDocument) (tasks.RelevanceScores, error) {
	return s.call.Call(ctx, recipe, pairs)
}

// Relevance prepares a reranker binding. Inputs are scored whole: the
// deployment's input policy must reject inputs over budget, never truncate them.
func (r *Runtime) Relevance(ctx context.Context, spec config.ResolvedModelBinding) (_ *RelevanceScorer, callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	input := spec.Deployment.WithDefaults().Input
	if spec.Binding.Contract != config.RelevanceScoresContract || input.Overflow != "reject" {
		return nil, fmt.Errorf("%w: a relevance scorer needs the %s contract and the reject input policy", binding.ErrCapability, config.RelevanceScoresContract)
	}
	ctx, cancel := preparationContext(ctx)
	defer cancel()
	card, err := r.card(ctx, spec)
	if err != nil {
		return nil, err
	}
	if !card.Serves("rerank") || card.Rerank == nil {
		return nil, fmt.Errorf("%w: deployment %q does not serve rerank", binding.ErrCapability, spec.Binding.Deployment)
	}
	selection, err := pairScorerExit(card, spec.Binding.PairScorer)
	if err != nil {
		return nil, err
	}
	task, err := r.relevanceTask()
	if err != nil {
		return nil, err
	}
	resource, err := r.acquire(ctx, spec, card)
	if err != nil {
		return nil, err
	}
	capability := binding.Capability{
		Contract: spec.Binding.Contract, Provider: Provider, Device: card.Device, Precision: card.Dtype,
		Limits: binding.Limits{ModelTokens: card.MaxInputTokens, DeploymentTokens: input.MaxTokens, Overflow: input.Overflow},
	}
	deployment, services := spec.Binding.Deployment, r.services
	request := modelservice.RerankRequest{Layer: selection.Layer, Dimensions: selection.Dimension, Overflow: input.Overflow, MaxTokens: input.MaxTokens}
	infer := func(ctx context.Context, _ io.Closer, pairs []tasks.QueryDocument) (tasks.RelevanceScores, error) {
		return rerankPairs(ctx, services, deployment, request, pairs)
	}
	t := &target{spec: spec, deployment: deployment, card: card, resource: resource}
	call, err := publish(ctx, task, t, capability, infer, []tasks.QueryDocument{{Query: "warmup", Document: "warmup"}})
	if err != nil {
		return nil, err
	}
	descriptor, _ := json.Marshal(struct {
		Numerics  string
		Selection config.PairScorerSelection
		Semantics tasks.ScoreSemantics
	}{numerics(card, capability.Limits), selection, tasks.RelevanceScoreSemantics()})
	digest := sha256.Sum256(descriptor)
	return &RelevanceScorer{call: call, selection: selection, identity: hex.EncodeToString(digest[:])}, nil
}

// pairScorerExit resolves the binding's exit against the card: without a
// selection the deployment's default exit, otherwise the first declared exit
// (the default first) matching the selected layer and dimension.
func pairScorerExit(card modelservice.ModelCard, requested *config.PairScorerSelection) (config.PairScorerSelection, error) {
	wanted := config.PairScorerSelection{}
	if requested != nil {
		wanted = *requested
	}
	if wanted.Layer < 0 || wanted.Dimension < 0 {
		return wanted, fmt.Errorf("%w: negative pair-scorer selection", binding.ErrCapability)
	}
	for _, exit := range append([]modelservice.RerankExit{card.Rerank.Default}, card.Rerank.Exits...) {
		if (wanted.Layer == 0 || wanted.Layer == exit.Layer) && (wanted.Dimension == 0 || wanted.Dimension == exit.Dimension) {
			return config.PairScorerSelection{Layer: exit.Layer, Dimension: exit.Dimension}, nil
		}
	}
	return wanted, fmt.Errorf("%w: model %q has no pair-scorer exit at layer %d, dimension %d", binding.ErrCapability, card.ID, wanted.Layer, wanted.Dimension)
}

func (r *Runtime) relevanceTask() (*binding.Task[[]tasks.QueryDocument, tasks.RelevanceScores], error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if task, err := binding.Lookup[[]tasks.QueryDocument, tasks.RelevanceScores](r.registry, config.RelevanceScoresContract); err == nil {
		return task, nil
	}
	return binding.Register(r.registry, config.RelevanceScoresContract, validatePairs, tasks.ValidateRelevanceScores)
}

func validatePairs(pairs []tasks.QueryDocument) error {
	if len(pairs) == 0 {
		return fmt.Errorf("at least one query/document pair is required")
	}
	for _, pair := range pairs {
		if !utf8.ValidString(pair.Query) || !utf8.ValidString(pair.Document) {
			return fmt.Errorf("query and document must be valid UTF-8")
		}
		if err := validateText(pair.Query); err != nil {
			return err
		}
		if err := validateText(pair.Document); err != nil {
			return err
		}
	}
	return nil
}

// rerankPairs sends one rerank call per run of pairs that share a query and
// returns the logits in input order.
func rerankPairs(ctx context.Context, services Services, deployment string, template modelservice.RerankRequest, pairs []tasks.QueryDocument) (tasks.RelevanceScores, error) {
	scores := tasks.RelevanceScores{Scores: make([]float32, len(pairs)), Inputs: make([]tasks.InputUsage, len(pairs))}
	for start := 0; start < len(pairs); {
		end := start + 1
		for end < len(pairs) && pairs[end].Query == pairs[start].Query {
			end++
		}
		request := template
		request.Query = pairs[start].Query
		request.Documents = make([]string, 0, end-start)
		for _, pair := range pairs[start:end] {
			request.Documents = append(request.Documents, pair.Document)
		}
		response, err := services.Rerank(ctx, deployment, request)
		if err != nil {
			return tasks.RelevanceScores{}, err
		}
		for i, result := range response.Results {
			if result.Error != "" {
				return tasks.RelevanceScores{}, itemError(result.Error)
			}
			scores.Scores[start+i] = float32(result.Logit)
			if usage := inputUsage(result.Input); usage != nil {
				scores.Inputs[start+i] = *usage
			}
		}
		start = end
	}
	return scores, nil
}

// DiagnoseRerank runs a recipe's prepared relevance binding on pairs.
func (r *Runtime) DiagnoseRerank(ctx context.Context, recipe, name string, pairs []tasks.QueryDocument) (DiagnosticResult[tasks.RelevanceScores], error) {
	var response DiagnosticResult[tasks.RelevanceScores]
	handle, metadata, err := binding.LookupPrepared[[]tasks.QueryDocument, tasks.RelevanceScores](r.prepared, recipe, name, config.RelevanceScoresContract)
	if err != nil {
		return response, err
	}
	result, err := handle.Call(ctx, recipe, pairs)
	return DiagnosticResult[tasks.RelevanceScores]{Binding: metadata, Result: result}, err
}
