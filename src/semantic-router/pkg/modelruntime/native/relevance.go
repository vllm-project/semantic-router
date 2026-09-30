package native

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
)

// RelevanceScorer owns a recipe's reference to a fixed native exit. Native
// resources share the existing pool; a request cannot select another head.
type RelevanceScorer struct {
	task      *binding.Resolved[[]tasks.QueryDocument, tasks.RelevanceScores]
	selection config.PairScorerSelection
	identity  string
}

func (s *RelevanceScorer) Close() error                          { return s.task.Close() }
func (s *RelevanceScorer) CacheIdentity() string                 { return s.identity }
func (s *RelevanceScorer) Selection() config.PairScorerSelection { return s.selection }
func (s *RelevanceScorer) ScorePairs(ctx context.Context, recipe string, pairs []tasks.QueryDocument) (tasks.RelevanceScores, error) {
	return s.task.Call(ctx, recipe, pairs)
}

func (r *Runtime) relevanceTask() (*binding.Task[[]tasks.QueryDocument, tasks.RelevanceScores], error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if task, err := binding.Lookup[[]tasks.QueryDocument, tasks.RelevanceScores](r.registry, config.RelevanceScoresContract); err == nil {
		return task, nil
	}
	return binding.Register(r.registry, config.RelevanceScoresContract, func(pairs []tasks.QueryDocument) error {
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
	}, tasks.ValidateRelevanceScores)
}

// Relevance prepares and warms a complete encoder/head artifact. Artifact
// fingerprints include encoder, task heads, tokenizer and mathematical config.
// A Runtime memoizes these fingerprints and treats its artifacts as immutable;
// replacing bytes requires a new Runtime generation. Ranking caches are local
// to that router process, rather than persisted across software releases.
func (r *Runtime) Relevance(ctx context.Context, spec config.ResolvedModelBinding) (_ *RelevanceScorer, callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	if spec.Binding.Contract != config.RelevanceScoresContract || spec.Binding.Adapter != "vela_reranker" || spec.Deployment.WithDefaults().Input.Overflow != "reject" {
		return nil, fmt.Errorf("%w: relevance scorer requires its raw-logit adapter and reject input policy", binding.ErrCapability)
	}
	selection := config.PairScorerSelection{}
	if spec.Binding.PairScorer != nil {
		selection = *spec.Binding.PairScorer
	}
	if selection.Layer < 0 || selection.Dimension < 0 {
		return nil, fmt.Errorf("%w: negative scorer selection", binding.ErrCapability)
	}
	task, err := r.relevanceTask()
	if err != nil {
		return nil, err
	}
	var prepared *preparedRelevance
	switch spec.Deployment.Provider {
	case "candle":
		prepared, err = r.candleRelevance(ctx, spec, selection)
	case "ort":
		prepared, err = r.ortRelevance(ctx, spec, selection)
	default:
		err = fmt.Errorf("%w: relevance provider unavailable", binding.ErrCapability)
	}
	if err != nil {
		return nil, err
	}
	actual := prepared.selection
	resource := prepared.resource
	capability := prepared.capability
	if actual.Layer <= 0 || actual.Dimension <= 0 || (selection.Layer != 0 && selection.Layer != actual.Layer) || (selection.Dimension != 0 && selection.Dimension != actual.Dimension) {
		_ = resource.Close()
		return nil, fmt.Errorf("%w: loaded pair scorer selection differs from its binding", binding.ErrCapability)
	}
	// Content and effective semantics separate cached ranking from a prior
	// generation at the same path. Artifact hashes are captured during load.
	revision, err := r.artifactRevision(ctx, spec.Deployment.Artifact)
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	descriptor, err := json.Marshal(struct {
		Version                         int
		Artifact, HeadArtifact, Adapter string
		Selection                       config.PairScorerSelection
		MaxTokens                       int
		Overflow                        string
		Semantics                       tasks.ScoreSemantics
		Execution                       []relevanceExecution
	}{1, revision, prepared.headRevision, spec.Binding.Adapter, actual, capability.Limits.EffectiveTokens(), capability.Limits.Overflow, tasks.RelevanceScoreSemantics(), prepared.execution})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	bound, err := finishNativeTask(ctx, spec, task, capability, resource, prepared.infer, []tasks.QueryDocument{{Query: "warmup", Document: "warmup"}})
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256(descriptor)
	return &RelevanceScorer{task: bound, selection: actual, identity: hex.EncodeToString(digest[:])}, nil
}

// A provider prepares its owned resource together with the effective model
// contract and execution identity. Task publication never inspects native types.
type preparedRelevance struct {
	resource     *binding.Resource
	capability   binding.Capability
	selection    config.PairScorerSelection
	headRevision string
	execution    []relevanceExecution
	infer        func(context.Context, io.Closer, []tasks.QueryDocument) (tasks.RelevanceScores, error)
}

// Observational IDs, profiling paths and counters do not identify model math.
// Artifact fingerprints bind relative names and bytes, not bare file SHA256.
// Candle exposes a math contract version, not an attested runtime build ID.
type relevanceExecution struct {
	GraphFingerprint, ExternalArtifactFingerprint string
	CompilerFlags                                 map[string]string
	MathContract                                  string
	Provider, Device, Precision, RuntimeBuild     string
	CustomOpsProfile, CustomOpsSHA256             string
	CPUFallbackDisabled                           bool
}
