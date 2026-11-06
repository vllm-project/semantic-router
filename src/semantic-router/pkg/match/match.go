// Package match runs the match block (config.MatchConfig) of the features
// that find similar items. Recall narrows a feature's candidates by embedding
// or BM25 similarity, through the feature's own index or, over a candidate
// list, the lexical index of pkg/vectorstore. Judge asks one System One
// question about the request and the candidates, in one call, compiled for
// the model family of the deployment that answers it, and keeps the
// candidates by their scores.
package match

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

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

// Input is what a match scores: the request, and the candidates, in an index
// or listed. Recall reads the Index when there is one, and otherwise ranks
// the listed Candidates by BM25; a judge without recall reads the listed
// Candidates.
type Input struct {
	Request string
	// History is the earlier user turns a judge reads beside the request,
	// for a feature that reads them; empty reads the request alone.
	History    string
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

// ErrTooManyCandidates is a choice judge given more candidates than one
// question takes options.
var ErrTooManyCandidates = errors.New("a choice judge chooses among at most 255 candidates")

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
// that fails, and a judge that fails with on_error fail, fail the match; the
// result still reports every stage that ran.
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
	if err := distinctIDs(len(input.Candidates), func(i int) string { return input.Candidates[i].ID }); err != nil {
		return result, err
	}
	candidates := make([]Scored, len(input.Candidates))
	for index, candidate := range input.Candidates {
		candidates[index] = Scored{Candidate: candidate}
	}
	if spec.Recall != nil {
		recalled, stage, err := recall(ctx, *spec.Recall, input)
		result.record(feature, stage)
		if err != nil {
			return result, err
		}
		candidates = recalled
	} else if input.Index != nil {
		return result, errors.New("match: a judge cannot read an index whole; recall from it")
	}
	if spec.Judge == nil {
		result.Kept = candidates
		return result, nil
	}
	judged, stage, err := m.judge(ctx, *spec.Judge, input, candidates)
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
	if errors.Is(err, ErrTooManyCandidates) {
		return "too_many_candidates"
	}
	return modelservice.ErrorReason(err)
}
