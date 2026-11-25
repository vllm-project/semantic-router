// Package match runs the match block (config.MatchConfig) of the features
// that find similar items. Recall narrows a feature's candidates by embedding
// or BM25 similarity, through the feature's own index or, over a candidate
// list, the lexical index of pkg/vectorstore. Judge asks one System One
// question about the request and the candidates, in one call, compiled for
// the model family of the deployment that answers it, and keeps the
// candidates by their scores. A judge reads at most its max_candidates
// candidates and truncates the rest.
package match

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// Answerer asks System One questions of the deployments judges name: Decide
// sends a deployment's questions, and Card is the card they compile against,
// false while the deployment has none ready. A modelservice.Remote is one for
// the systemone deployments it serves.
type Answerer interface {
	modelservice.Decider
	Card(deployment string) (modelservice.ModelCard, bool)
}

// Candidate is one item a match scores: the ID its feature knows it by, and
// the text recall compares with the request and the judge reads.
type Candidate struct {
	ID   string
	Text string
}

// Scored is a candidate with its scores.
type Scored struct {
	Candidate
	// Similarity is recall's, zero when recall did not run.
	Similarity float64
	// Score orders the candidates: the judge's, else the similarity.
	Score float64
}

// Index recalls a feature's candidates from its store: the cached requests,
// the documents of a vector store, a user's memories.
type Index interface {
	// Recall returns the candidates most similar to query by recall's
	// method, at most recall.TopK, each at least recall.MinSimilarity
	// similar, with their similarity.
	Recall(ctx context.Context, query string, recall config.MatchRecall) ([]Scored, error)
}

// Lister is an Index that a judge without recall reads whole: a small store,
// such as a user's memories or a local knowledge base.
type Lister interface {
	// List returns at most limit of the store's candidates, most recent
	// first, and how many it holds.
	List(ctx context.Context, limit int) (candidates []Candidate, held int, err error)
}

// Input is what a match scores: the request, and the candidates, in an index
// or listed. Recall reads the Index when there is one, and otherwise ranks
// the listed Candidates by BM25; a judge without recall reads the listed
// Candidates, or the ones a Lister index lists.
type Input struct {
	Request string
	// History is the earlier user turns a judge reads beside the request,
	// for a feature that reads them; empty reads the request alone.
	History string
	// Candidates are listed in the order a judge without recall reads the
	// first of when they are more than its max_candidates: most recent
	// first, where the feature's items have an age.
	Candidates []Candidate
	Index      Index
}

// State is the state a choice judge asks about: the request, with the
// earlier turns as their own part when the input carries them. A caller's
// question about the same state, asked in the judge's bundle, shares its
// call.
func (input Input) State() modelservice.Request {
	if strings.TrimSpace(input.History) == "" {
		return modelservice.Request{State: input.Request}
	}
	return modelservice.Request{Parts: map[string]string{historyPart: input.History, requestPart: input.Request}}
}

// Result is what a match kept, best first, and what each of its stages did.
type Result struct {
	Kept   []Scored
	Stages []Stage
}

// Stage reports one stage of a match, for its metrics and for the replay and
// trace records of the feature that ran it.
type Stage struct {
	// Name is recall or judge.
	Name string
	// Method is the recall method, or the judge's preset (question for a
	// custom one).
	Method string
	// Deployment answered the judge.
	Deployment string
	// Candidates is how many the stage scored, and Kept how many it kept.
	Candidates int
	Kept       int
	// Truncated is how many candidates over its max_candidates a judge
	// did not read.
	Truncated int
	// Outcome is ok, empty (nothing to score), skipped (a failed judge
	// whose on_error is skip) or failed; Reason says why it failed.
	Outcome  string
	Reason   string
	Duration time.Duration
}

// Stage names and outcomes.
const (
	StageRecall    = "recall"
	StageJudge     = "judge"
	OutcomeOK      = "ok"
	OutcomeEmpty   = "empty"
	OutcomeSkipped = "skipped"
	OutcomeFailed  = "failed"
)

// Matcher runs the match blocks of one configuration.
type Matcher struct {
	cfg      *config.RouterConfig
	answerer Answerer
}

// New returns a Matcher whose judges ask the deployments cfg resolves
// through answerer.
func New(cfg *config.RouterConfig, answerer Answerer) *Matcher {
	return &Matcher{cfg: cfg, answerer: answerer}
}

// Match runs the block a feature writes, its default when block is nil, over
// input. A judge whose on_error is skip and that fails keeps the recall
// result, or the listed candidates in their order without recall. A recall
// that fails, a Lister that fails to list, and a judge that fails with
// on_error fail, fail the match; the result still reports every stage that
// ran.
//
// A judge reads at most its max_candidates candidates: the first by recall
// rank, or as listed without recall. It never refuses more; each truncation
// is an event on the request's trace span and a count in a metric.
//
// A judge asks about every candidate in one call. Inside a modelservice
// Bundle it asks as a participant of the caller's, so its call also carries
// the caller's other questions to the deployment about the same state: a
// choice judge's state is the request, modelservice.Request{State: request}.
func (m *Matcher) Match(ctx context.Context, feature config.MatchFeature, block *config.MatchConfig, input Input) (Result, error) {
	spec := feature.Effective(block)
	var result Result
	if strings.TrimSpace(input.Request) == "" {
		return result, errors.New("match: the request is empty")
	}
	candidates, held, err := m.candidates(ctx, feature, spec, input, &result)
	if err != nil {
		return result, err
	}
	if spec.Judge == nil {
		result.Kept = candidates
		return result, nil
	}
	read := candidates[:min(len(candidates), spec.Judge.EffectiveMaxCandidates())]
	judged, stage, err := m.judge(ctx, *spec.Judge, input, read)
	if stage.Truncated = held - len(read); stage.Truncated > 0 {
		annotateTruncation(ctx, feature, stage, spec.Recall != nil)
	}
	if err != nil {
		stage.Outcome = OutcomeFailed
		if spec.Judge.EffectiveOnError() == config.MatchOnErrorSkip {
			stage.Outcome, judged, err = OutcomeSkipped, candidates, nil
		}
	}
	result.record(feature, stage)
	result.Kept = judged
	return result, err
}

// candidates are what a match's judge reads the first of, or its result
// without a judge: the recalled candidates, best first, or the listed ones,
// and how many there are, counting those a Lister held back.
func (m *Matcher) candidates(ctx context.Context, feature config.MatchFeature, spec config.MatchConfig, input Input, result *Result) ([]Scored, int, error) {
	listed := input.Candidates
	switch {
	case spec.Recall != nil:
		if err := distinctIDs(len(listed), func(i int) string { return listed[i].ID }); err != nil {
			return nil, 0, err
		}
		recalled, stage, err := recall(ctx, *spec.Recall, input)
		result.record(feature, stage)
		return recalled, len(recalled), err
	case input.Index != nil:
		lister, ok := input.Index.(Lister)
		if !ok || spec.Judge == nil {
			return nil, 0, errors.New("match: a judge cannot read this index whole; recall from it")
		}
		var held int
		var err error
		if listed, held, err = lister.List(ctx, spec.Judge.EffectiveMaxCandidates()); err != nil {
			return nil, 0, fmt.Errorf("match: listing the candidates failed: %w", err)
		}
		if err := distinctIDs(len(listed), func(i int) string { return listed[i].ID }); err != nil {
			return nil, 0, err
		}
		return asScored(listed), max(held, len(listed)), nil
	}
	if err := distinctIDs(len(listed), func(i int) string { return listed[i].ID }); err != nil {
		return nil, 0, err
	}
	return asScored(listed), len(listed), nil
}

func asScored(candidates []Candidate) []Scored {
	scored := make([]Scored, len(candidates))
	for index, candidate := range candidates {
		scored[index] = Scored{Candidate: candidate}
	}
	return scored
}

// annotateTruncation records a judge that read only its first max_candidates
// candidates, by recall rank or as listed: an event on the request's span,
// and a count in vsr_match_judge_truncated_total.
func annotateTruncation(ctx context.Context, feature config.MatchFeature, stage Stage, byRecall bool) {
	order := "listed"
	if byRecall {
		order = "recall_rank"
	}
	truncatedTotal.WithLabelValues(string(feature), stage.Method).Inc()
	trace.SpanFromContext(ctx).AddEvent("match.judge.truncated", trace.WithAttributes(
		attribute.String("match.feature", string(feature)),
		attribute.String("match.judge.method", stage.Method),
		attribute.Int("match.judge.candidates", stage.Candidates),
		attribute.Int("match.judge.truncated", stage.Truncated),
		attribute.String("match.judge.order", order),
	))
}

func (r *Result) record(feature config.MatchFeature, stage Stage) {
	observe(feature, stage)
	r.Stages = append(r.Stages, stage)
}

// distinctIDs refuses candidates a match could not tell apart.
func distinctIDs(count int, id func(int) string) error {
	seen := make(map[string]bool, count)
	for index := range count {
		if id(index) == "" || seen[id(index)] {
			return fmt.Errorf("match: candidate IDs must be nonempty and distinct; %q is not", id(index))
		}
		seen[id(index)] = true
	}
	return nil
}

// reason is the low-cardinality reason a stage failed.
func reason(err error) string {
	var failed answerFailure
	if errors.As(err, &failed) {
		return string(failed)
	}
	return modelservice.ErrorReason(err)
}
