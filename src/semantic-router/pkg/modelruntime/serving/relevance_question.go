package serving

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

const (
	decisionRelevanceTask          = "rag_relevance.decision.v1"
	decisionRelevanceMaxCandidates = config.MaxDecisionRAGCandidates
	decisionRelevanceMaxBytes      = 1 << 20
	decisionRelevanceTimeout       = 30 * time.Second
)

func (r *Runtime) decisionRelevance(ctx context.Context, spec config.ResolvedModelBinding) (*RelevanceScorer, error) {
	if spec.Name != config.RAGRerankerConsumer || spec.Binding.Adapter != "" || spec.Binding.PairScorer != nil || spec.Binding.Head != "" || spec.Binding.MappingPath != "" || spec.Binding.OperatingPoint != nil {
		return nil, fmt.Errorf("%w: decision relevance requires an unmodified rag.reranker decision binding", binding.ErrCapability)
	}
	ctx, cancel := preparationContext(ctx)
	defer cancel()
	card, err := r.card(ctx, spec)
	if err != nil {
		return nil, err
	}
	decider, ok := r.services.(modelservice.Decider)
	if !ok {
		return nil, fmt.Errorf("%w: deployment cannot answer decisions", binding.ErrCapability)
	}
	definition, _ := modelservice.BuiltinTask("rag_relevance")
	if _, err = modelservice.CompileTask(definition, definition.Question, card); err != nil {
		return nil, err
	}
	scan, err := questionScanBudget(spec, card)
	if err != nil {
		return nil, err
	}
	task, err := r.decisionRelevanceTask()
	if err != nil {
		return nil, err
	}
	resource, err := r.acquire(ctx, spec, card)
	if err != nil {
		return nil, err
	}
	capability := binding.Capability{
		Contract: config.DecisionTaskContract, Provider: Provider, Device: card.Device, Precision: card.Dtype,
		Question: definition.ID, Deployment: spec.Binding.Deployment,
		Limits: binding.Limits{ModelTokens: card.MaxInputTokens, Overflow: spec.Deployment.WithDefaults().Input.Overflow},
	}
	target := &target{spec: spec, deployment: spec.Binding.Deployment, card: card, resource: resource, scan: scan}
	infer := func(ctx context.Context, _ io.Closer, pairs []tasks.QueryDocument) (tasks.RelevanceScores, error) {
		return scoreDecisionPairs(ctx, decider, target.deployment, card, definition.Question, scan, pairs)
	}
	call, err := publish(ctx, task, target, capability, infer, []tasks.QueryDocument{{Query: "When is the library open?", Document: "The library opens on weekdays."}})
	if err != nil {
		return nil, err
	}
	identity := ""
	// Without immutable model content, retrieval caching cannot be isolated.
	if card.ModelSHA256 != "" {
		data, _ := json.Marshal(struct {
			Numerics string
			Question modelservice.Question
			Scan     int
			Contract string
		}{numerics(card, capability.Limits), definition.Question, scan, decisionRelevanceTask})
		digest := sha256.Sum256(data)
		identity = hex.EncodeToString(digest[:])
	}
	return &RelevanceScorer{call: call, identity: identity}, nil
}

func (r *Runtime) decisionRelevanceTask() (*binding.Task[[]tasks.QueryDocument, tasks.RelevanceScores], error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if task, err := binding.Lookup[[]tasks.QueryDocument, tasks.RelevanceScores](r.registry, decisionRelevanceTask); err == nil {
		return task, nil
	}
	return binding.RegisterTask(r.registry, decisionRelevanceTask, config.DecisionTaskContract, validateDecisionPairs, tasks.ValidateRelevanceScores)
}

func validateDecisionPairs(pairs []tasks.QueryDocument) error {
	if len(pairs) > decisionRelevanceMaxCandidates {
		return fmt.Errorf("decision relevance accepts at most %d candidates", decisionRelevanceMaxCandidates)
	}
	bytes := 0
	for _, pair := range pairs {
		if len(pair.Query) > decisionRelevanceMaxBytes-bytes {
			return fmt.Errorf("decision relevance input exceeds byte budget")
		}
		bytes += len(pair.Query)
		if len(pair.Document) > decisionRelevanceMaxBytes-bytes {
			return fmt.Errorf("decision relevance input exceeds byte budget")
		}
		bytes += len(pair.Document)
	}
	return validatePairs(pairs)
}

// Each candidate is a separate state. The existing bundle may coalesce their
// transport and compatible work; this does not promise a single model forward.
func scoreDecisionPairs(ctx context.Context, decider modelservice.Decider, deployment string, card modelservice.ModelCard, question modelservice.Question, scan int, pairs []tasks.QueryDocument) (tasks.RelevanceScores, error) {
	ctx, cancel := context.WithTimeout(ctx, decisionRelevanceTimeout)
	defer cancel()
	ctx, bundle := modelservice.WithBundle(ctx, modelservice.DefaultBundleWindow)
	result := tasks.RelevanceScores{Scores: make([]float32, len(pairs)), Decision: &tasks.DecisionRelevanceEvidence{Coverage: make([]string, len(pairs))}}
	responses := make([]modelservice.Response, len(pairs))
	failures := make([]error, len(pairs))
	contexts := make([]context.Context, len(pairs))
	leaves := make([]func(), len(pairs))
	for i := range pairs {
		contexts[i], leaves[i] = bundle.JoinAsking(ctx, deployment)
	}
	var workers sync.WaitGroup
	for i, pair := range pairs {
		workers.Add(1)
		go func(i int, pair tasks.QueryDocument) {
			defer workers.Done()
			defer leaves[i]()
			asked := question
			asked.ID = fmt.Sprintf("rag.reranker:candidate:%d", i)
			responses[i], failures[i] = modelservice.ExecuteQuestions(contexts[i], exactRelevanceAnswers{decider}, deployment, card, modelservice.Request{
				Parts: map[string]string{"request": pair.Query, "context": pair.Document}, Questions: []modelservice.Question{asked}, MaxTokens: scan,
			})
			if failures[i] != nil {
				return
			}
			answer := responses[i].Answers[asked.ID]
			if answer.Error != "" {
				failures[i] = itemError(answer.Error)
				return
			}
			result.Scores[i] = float32(answer.Score)
			result.Decision.Coverage[i] = answer.InputCoverage
		}(i, pair)
	}
	workers.Wait()
	for i, err := range failures {
		if err != nil {
			return tasks.RelevanceScores{}, fmt.Errorf("candidate %d: %w", i, err)
		}
		result.Decision.InputTokens += responses[i].InputTokens
	}
	if err := ctx.Err(); err != nil {
		return tasks.RelevanceScores{}, err
	}
	return result, nil
}

// Check the answer set before ExecuteQuestions reduces it to declared keys.
// Document identity stays in the caller, never in model-generated text.
type exactRelevanceAnswers struct{ modelservice.Decider }

func (d exactRelevanceAnswers) Decide(ctx context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	response, err := d.Decider.Decide(ctx, deployment, request)
	if err != nil {
		return response, err
	}
	if len(response.Answers) != len(request.Questions) {
		return response, fmt.Errorf("%w: relevance answer count differs from candidates", binding.ErrInvalidResult)
	}
	for _, question := range request.Questions {
		if _, ok := response.Answers[question.ID]; !ok {
			return response, fmt.Errorf("%w: relevance answer has an undeclared candidate", binding.ErrInvalidResult)
		}
	}
	return response, nil
}
