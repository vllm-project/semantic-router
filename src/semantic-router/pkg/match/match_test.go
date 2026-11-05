package match

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// judgeModel is a fake decision model: a question about a candidate is
// answered with that candidate's score (P(yes), or the level of a score
// question), and a choice with the distribution over its options. It records
// every state it is asked about, and can fail, stall or fail one answer.
type judgeModel struct {
	mu        sync.Mutex
	family    string
	noCard    bool
	scores    map[string]float64
	choose    map[string]float64
	asked     []modelservice.Request
	err       error
	stall     bool
	answerErr string
}

func (j *judgeModel) Card(string) (modelservice.ModelCard, bool) {
	return modelservice.ModelCard{ID: "judge", Family: j.family, Surfaces: []string{"decisions"}}, !j.noCard
}

func (j *judgeModel) Decide(ctx context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	j.mu.Lock()
	j.asked = append(j.asked, request)
	j.mu.Unlock()
	if j.stall {
		<-ctx.Done()
		return modelservice.Response{}, ctx.Err()
	}
	if j.err != nil {
		return modelservice.Response{}, j.err
	}
	answers := make(map[string]modelservice.Answer, len(request.Questions))
	for _, question := range request.Questions {
		answer := modelservice.Answer{Type: question.Type, Error: j.answerErr}
		switch question.Type {
		case config.SystemOneQuestionNoul:
			answer.Noul = j.scores[request.Parts[candidatePart]]
		case config.SystemOneQuestionScore:
			answer.Score = j.scores[request.Parts[candidatePart]]
			answer.Probabilities = levelDistribution(len(question.Levels), answer.Score)
		case config.SystemOneQuestionChoice:
			answer.Probabilities = map[string]float64{}
			for _, choice := range question.Choices {
				answer.Probabilities[choice.Key] = j.choose[choice.Key]
				if answer.Choice == "" || j.choose[choice.Key] > j.choose[answer.Choice] {
					answer.Choice = choice.Key
				}
			}
		}
		answers[question.ID] = answer
	}
	return modelservice.Response{Answers: answers}, nil
}

// levelDistribution puts the mass of an expected level on the two levels
// around it.
func levelDistribution(levels int, expected float64) map[string]float64 {
	distribution := make(map[string]float64, levels)
	for level := range levels {
		distribution[strconv.Itoa(level)] = 0
	}
	low := int(expected)
	distribution[strconv.Itoa(low)] = 1 - (expected - float64(low))
	if high := low + 1; high < levels {
		distribution[strconv.Itoa(high)] = expected - float64(low)
	}
	return distribution
}

func (j *judgeModel) states() []modelservice.Request {
	j.mu.Lock()
	defer j.mu.Unlock()
	return append([]modelservice.Request(nil), j.asked...)
}

// index is a fake store: it returns its candidates as they are, whatever the
// recall's bounds, so the match must hold them.
type index struct {
	recalled []Scored
	err      error
	asked    []config.MatchRecall
}

func (x *index) Recall(_ context.Context, _ string, recall config.MatchRecall) ([]Scored, error) {
	x.asked = append(x.asked, recall)
	return x.recalled, x.err
}

func matchConfig() *config.RouterConfig {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		config.DefaultDecisionDeployment: {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/vela3-fast"},
		"remote":                         {Provider: config.SystemOneProvider, Endpoint: "https://api.example.com/v1/systemone", Model: "vllm-sr/d3-ultra", TimeoutMs: 40},
	}
	return cfg
}

func listed(texts ...string) []Candidate {
	candidates := make([]Candidate, len(texts))
	for index, text := range texts {
		candidates[index] = Candidate{ID: fmt.Sprintf("c%d", index), Text: text}
	}
	return candidates
}

func recalled(similarities ...float64) []Scored {
	candidates := make([]Scored, len(similarities))
	for index, similarity := range similarities {
		candidates[index] = Scored{Candidate: Candidate{ID: fmt.Sprintf("d%d", index), Text: fmt.Sprintf("doc %d", index)}, Similarity: similarity}
	}
	return candidates
}

func ids(scored []Scored) []string {
	keys := make([]string, len(scored))
	for index, candidate := range scored {
		keys[index] = candidate.ID
	}
	return keys
}

func bound(value float64) *float64 { return &value }

// Recall alone is classic similarity: the store's candidates, held to the
// recall's bounds, most similar first, scored by similarity. No model is
// asked.
func TestRecallOnlyKeepsTheMostSimilarWithinItsBounds(t *testing.T) {
	model := &judgeModel{}
	store := &index{recalled: recalled(0.75, 0.95, 0.7, 0.85, 0.9)}
	result, err := New(matchConfig(), model).Match(context.Background(), config.MatchResponseCache, nil, Input{Request: "reset my password", Index: store})
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(result.Kept); !reflect.DeepEqual(got, []string{"d1", "d4", "d3"}) {
		t.Fatalf("kept %v: the default cache recall keeps the 3 most similar at 0.8 or more", got)
	}
	if result.Kept[0].Score != 0.95 || result.Kept[0].Similarity != 0.95 {
		t.Fatalf("a recalled candidate is scored by its similarity: %+v", result.Kept[0])
	}
	if want := (config.MatchRecall{Method: config.MatchRecallEmbedding, TopK: 3, MinSimilarity: bound(0.8)}); !reflect.DeepEqual(store.asked, []config.MatchRecall{want}) {
		t.Fatalf("the store recalls with the feature's bounds: %+v", store.asked)
	}
	if len(model.states()) != 0 || len(result.Stages) != 1 || result.Stages[0].Name != StageRecall || result.Stages[0].Outcome != OutcomeOK || result.Stages[0].Candidates != 5 || result.Stages[0].Kept != 3 {
		t.Fatalf("stages %+v, %d model calls", result.Stages, len(model.states()))
	}
}

// BM25 recall over listed candidates ranks every one of them with the
// lexical index, the ones without a query term last.
func TestBM25RecallRanksListedCandidates(t *testing.T) {
	block := &config.MatchConfig{Recall: &config.MatchRecall{Method: config.MatchRecallBM25, TopK: 3}}
	result, err := New(matchConfig(), &judgeModel{}).Match(context.Background(), config.MatchCompress, block, Input{
		Request:    "rotate the signing key",
		Candidates: listed("weather today", "rotate the signing key every month", "the signing key lives in the vault", "lunch menu"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(result.Kept); !reflect.DeepEqual(got, []string{"c1", "c2", "c0"}) {
		t.Fatalf("kept %v", got)
	}
	if result.Kept[0].Similarity <= result.Kept[1].Similarity || result.Kept[2].Similarity != 0 {
		t.Fatalf("scores %+v", result.Kept)
	}
	embedding := &config.MatchConfig{Recall: &config.MatchRecall{TopK: 3}}
	if _, err := New(matchConfig(), &judgeModel{}).Match(context.Background(), config.MatchCompress, embedding, Input{Request: "x", Candidates: listed("a")}); err == nil {
		t.Fatal("embedding recall over listed candidates has no embeddings to compare")
	}
}

// A judge alone asks about each listed candidate, one state each with the
// request and the candidate as parts, and keeps them by score.
func TestJudgeOnlyAsksAboutEachCandidate(t *testing.T) {
	model := &judgeModel{scores: map[string]float64{"error budget policy": 0.9, "release train": 0.2, "paging rules": 0.6}}
	result, err := New(matchConfig(), model).Match(context.Background(), config.MatchCompress, nil, Input{
		Request:    "when do we page the on-call engineer?",
		Candidates: listed("release train", "paging rules", "error budget policy"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(result.Kept); !reflect.DeepEqual(got, []string{"c2", "c1", "c0"}) {
		t.Fatalf("kept %v: compression keeps every segment, most relevant first", got)
	}
	relevance, _ := presets.Lookup(presets.Relevance, presets.StageRequest)
	states := model.states()
	if len(states) != 3 {
		t.Fatalf("%d states, want one per candidate", len(states))
	}
	for _, state := range states {
		question := state.Questions[0]
		if state.Parts["request"] != "when do we page the on-call engineer?" || state.Parts[candidatePart] == "" || len(state.Questions) != 1 ||
			question.ID != "match:relevance" || question.Type != config.SystemOneQuestionNoul || question.Instructions != relevance.Instructions || !question.Truncate {
			t.Fatalf("state %+v", state)
		}
	}
	if stage := result.Stages[0]; len(result.Stages) != 1 || stage.Name != StageJudge || stage.Method != presets.Relevance || stage.Deployment != config.DefaultDecisionDeployment || stage.Outcome != OutcomeOK || stage.Kept != 3 {
		t.Fatalf("stages %+v", result.Stages)
	}
}

// Recall narrows and the judge decides: RAG's default keeps the recalled
// candidates the relevance judge scores 0.5 or more, at most 6, each with
// its similarity.
func TestRecallThenJudge(t *testing.T) {
	store := &index{recalled: recalled(0.9, 0.8, 0.7, 0.6, 0.2)}
	model := &judgeModel{scores: map[string]float64{"doc 0": 0.3, "doc 1": 0.8, "doc 2": 0.5, "doc 3": 0.95}}
	result, err := New(matchConfig(), model).Match(context.Background(), config.MatchRAG, nil, Input{Request: "what is our refund window?", Index: store})
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(result.Kept); !reflect.DeepEqual(got, []string{"d3", "d1", "d2"}) {
		t.Fatalf("kept %v", got)
	}
	if result.Kept[0].Score != 0.95 || result.Kept[0].Similarity != 0.6 {
		t.Fatalf("a judged candidate keeps its similarity: %+v", result.Kept[0])
	}
	if len(model.states()) != 4 || len(result.Stages) != 2 || result.Stages[0].Kept != 4 || result.Stages[1].Candidates != 4 || result.Stages[1].Kept != 3 {
		t.Fatalf("the judge reads what recall kept: %d states, stages %+v", len(model.states()), result.Stages)
	}
}

func TestJudgeKeepRules(t *testing.T) {
	scores := map[string]float64{"a": 0.9, "b": 0.6, "c": 0.6, "d": 0.3, "e": 0.05}
	keepWith := func(edit func(*config.MatchJudge)) []string {
		judge := config.MatchJudge{Preset: presets.Relevance}
		edit(&judge)
		result, err := New(matchConfig(), &judgeModel{scores: scores}).Match(context.Background(), config.MatchCompress,
			&config.MatchConfig{Judge: &judge}, Input{Request: "q", Candidates: listed("e", "c", "a", "b", "d")})
		if err != nil {
			t.Fatal(err)
		}
		return ids(result.Kept)
	}
	cases := map[string]struct {
		edit func(*config.MatchJudge)
		want []string
	}{
		"every candidate, ties in their order": {func(*config.MatchJudge) {}, []string{"c2", "c1", "c3", "c4", "c0"}},
		"top_k":                                {func(j *config.MatchJudge) { j.TopK = 2 }, []string{"c2", "c1"}},
		"min_score":                            {func(j *config.MatchJudge) { j.MinScore = bound(0.6) }, []string{"c2", "c1", "c3"}},
		"min_ratio of the best":                {func(j *config.MatchJudge) { j.MinRatio = bound(0.5) }, []string{"c2", "c1", "c3"}},
		"all three":                            {func(j *config.MatchJudge) { j.MinScore, j.MinRatio, j.TopK = bound(0.1), bound(0.3), 3 }, []string{"c2", "c1", "c3"}},
		"none kept":                            {func(j *config.MatchJudge) { j.MinScore = bound(0.95) }, []string{}},
	}
	for name, test := range cases {
		if got := keepWith(test.edit); !reflect.DeepEqual(got, test.want) {
			t.Fatalf("%s: kept %v, want %v", name, got, test.want)
		}
	}
}

// tool_rank is one choice about the request, whose options are the
// candidates: one state, whatever their number, and each candidate scored
// by its probability.
func TestToolRankChoosesAmongTheCandidatesInOneState(t *testing.T) {
	model := &judgeModel{choose: map[string]float64{"search": 0.62, "calendar": 0.3, "billing": 0.05, "weather": 0.03}}
	tools := []Candidate{{ID: "weather", Text: "Current weather for a city"}, {ID: "search", Text: "Search the docs"}, {ID: "billing", Text: "Look up an invoice"}, {ID: "calendar", Text: "Book a meeting"}}
	result, err := New(matchConfig(), model).Match(context.Background(), config.MatchToolsSelect, nil, Input{Request: "find the onboarding guide", Candidates: tools})
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(result.Kept); !reflect.DeepEqual(got, []string{"search", "calendar"}) {
		t.Fatalf("kept %v: the 3 best at 0.1 of the best or more", got)
	}
	states := model.states()
	if len(states) != 1 || states[0].State != "find the onboarding guide" || states[0].Parts != nil || len(states[0].Questions) != 1 {
		t.Fatalf("states %+v", states)
	}
	question := states[0].Questions[0]
	want := []modelservice.Choice{{Key: "weather", Description: "Current weather for a city"}, {Key: "search", Description: "Search the docs"}, {Key: "billing", Description: "Look up an invoice"}, {Key: "calendar", Description: "Book a meeting"}}
	if question.ID != "match:tool_rank" || question.Type != config.SystemOneQuestionChoice || !reflect.DeepEqual(question.Choices, want) {
		t.Fatalf("question %+v", question)
	}
	if result.Kept[0].Score != 0.62 {
		t.Fatalf("scores %+v", result.Kept)
	}

	model = &judgeModel{}
	only, err := New(matchConfig(), model).Match(context.Background(), config.MatchToolsSelect, nil, Input{Request: "find the guide", Candidates: tools[:1]})
	if err != nil || len(only.Kept) != 1 || only.Kept[0].Score != 1 || len(model.states()) != 0 {
		t.Fatalf("one candidate is chosen without asking: %+v %v, %d calls", only.Kept, err, len(model.states()))
	}
	many := make([]Candidate, presets.MaxChoices+1)
	for index := range many {
		many[index] = Candidate{ID: strconv.Itoa(index)}
	}
	failing := &config.MatchConfig{Judge: &config.MatchJudge{Preset: presets.ToolRank, OnError: config.MatchOnErrorFail}}
	if _, err := New(matchConfig(), model).Match(context.Background(), config.MatchToolsSelect, failing, Input{Request: "q", Candidates: many}); !errors.Is(err, ErrTooManyCandidates) {
		t.Fatalf("err = %v", err)
	}
}

// A custom question: a score question about each candidate scores it with
// the expected level, and a choice question chooses among them.
func TestCustomJudgeQuestions(t *testing.T) {
	levels := []string{"unrelated", "related", "answers it"}
	model := &judgeModel{scores: map[string]float64{"a": 1.8, "b": 0.4, "c": 1.2}}
	score := &config.MatchConfig{Judge: &config.MatchJudge{Question: &config.SystemOneQuestion{Type: config.SystemOneQuestionScore, Instructions: "How well does the candidate answer?", Levels: levels}, MinScore: bound(1)}}
	result, err := New(matchConfig(), model).Match(context.Background(), config.MatchCompress, score, Input{Request: "q", Candidates: listed("b", "a", "c")})
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(result.Kept); !reflect.DeepEqual(got, []string{"c1", "c2"}) || result.Kept[0].Score != 1.8 {
		t.Fatalf("kept %+v", result.Kept)
	}
	if question := model.states()[0].Questions[0]; question.ID != "match:question" || !reflect.DeepEqual(question.Levels, levels) || result.Stages[0].Method != "question" {
		t.Fatalf("question %+v, stage %+v", question, result.Stages[0])
	}

	model = &judgeModel{choose: map[string]float64{"c0": 0.2, "c1": 0.8}}
	choose := &config.MatchConfig{Judge: &config.MatchJudge{Question: &config.SystemOneQuestion{Type: config.SystemOneQuestionChoice, Instructions: "Which passage answers the request?"}}}
	result, err = New(matchConfig(), model).Match(context.Background(), config.MatchCompress, choose, Input{Request: "q", Candidates: listed("first", "second")})
	if err != nil || !reflect.DeepEqual(ids(result.Kept), []string{"c1", "c0"}) || len(model.states()) != 1 {
		t.Fatalf("kept %+v %v, %d states", result.Kept, err, len(model.states()))
	}
}

// A preset is compiled for the answering deployment's family: a vela3_d3
// package answers a question it has no trained adapter for with its decision
// adapter.
func TestJudgeIsCompiledForTheDeploymentsFamily(t *testing.T) {
	for family, adapter := range map[string]bool{presets.FamilyVela3D3: true, presets.FamilyVela3: false, "decision3": false} {
		model := &judgeModel{family: family, scores: map[string]float64{"a": 0.9}}
		if _, err := New(matchConfig(), model).Match(context.Background(), config.MatchCompress, nil, Input{Request: "q", Candidates: listed("a")}); err != nil {
			t.Fatal(err)
		}
		if asked := model.states()[0].Questions[0]; asked.DecisionAdapter != adapter {
			t.Fatalf("%s: decision adapter %v", family, asked.DecisionAdapter)
		}
	}
}

// on_error skip keeps the recall result, or the listed candidates as they
// came; on_error fail fails the match. Either way the stage says why.
func TestJudgeFailures(t *testing.T) {
	store := func() *index { return &index{recalled: recalled(0.6, 0.9)} }
	rag := func(onError string, edit func(*config.MatchJudge)) *config.MatchConfig {
		block := config.MatchRAG.Default()
		block.Judge.OnError = onError
		if edit != nil {
			edit(block.Judge)
		}
		return &block
	}
	cases := map[string]struct {
		model  *judgeModel
		edit   func(*config.MatchJudge)
		reason string
	}{
		"runtime unavailable":  {&judgeModel{err: modelservice.ErrUnavailable}, nil, "unavailable"},
		"no card yet":          {&judgeModel{noCard: true}, nil, "unavailable"},
		"answer error":         {&judgeModel{answerErr: "invalid_model_output"}, nil, "invalid_model_output"},
		"timeout":              {&judgeModel{stall: true}, func(j *config.MatchJudge) { j.TimeoutMs = 20 }, "timeout"},
		"remote's own timeout": {&judgeModel{stall: true}, func(j *config.MatchJudge) { j.Deployment = "remote" }, "timeout"},
	}
	for name, test := range cases {
		started := time.Now()
		result, err := New(matchConfig(), test.model).Match(context.Background(), config.MatchRAG, rag(config.MatchOnErrorSkip, test.edit), Input{Request: "q", Index: store()})
		if err != nil || !reflect.DeepEqual(ids(result.Kept), []string{"d1", "d0"}) || result.Kept[0].Score != 0.9 {
			t.Fatalf("%s: skip keeps the recall result: %+v %v", name, result.Kept, err)
		}
		if stage := result.Stages[1]; stage.Outcome != OutcomeSkipped || stage.Reason != test.reason {
			t.Fatalf("%s: stage %+v", name, stage)
		}
		if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
			t.Fatalf("%s: a judge's deadline bounds it: %v", name, elapsed)
		}
		result, err = New(matchConfig(), test.model).Match(context.Background(), config.MatchRAG, rag(config.MatchOnErrorFail, test.edit), Input{Request: "q", Index: store()})
		if err == nil || result.Kept != nil || result.Stages[1].Outcome != OutcomeFailed || result.Stages[1].Reason != test.reason {
			t.Fatalf("%s: fail fails the match: %+v %v", name, result, err)
		}
	}
	timedOut, err := New(matchConfig(), &judgeModel{stall: true}).Match(context.Background(), config.MatchRAG, rag(config.MatchOnErrorFail, func(j *config.MatchJudge) { j.TimeoutMs = 10 }), Input{Request: "q", Index: store()})
	if !errors.Is(err, context.DeadlineExceeded) || len(timedOut.Stages) != 2 {
		t.Fatalf("a late judge fails with its deadline: %v", err)
	}
	unchanged, err := New(matchConfig(), &judgeModel{err: modelservice.ErrOverloaded}).Match(context.Background(), config.MatchCompress, nil, Input{Request: "q", Candidates: listed("x", "y", "z")})
	if err != nil || !reflect.DeepEqual(ids(unchanged.Kept), []string{"c0", "c1", "c2"}) || unchanged.Kept[0].Score != 0 {
		t.Fatalf("without recall, skip leaves the candidates as they came: %+v %v", unchanged.Kept, err)
	}
	if _, err := New(matchConfig(), &judgeModel{}).Match(context.Background(), config.MatchRAG, nil, Input{Request: "q", Index: &index{err: errors.New("vector store down")}}); err == nil {
		t.Fatal("a failed recall fails the match")
	}
}

func TestMatchWithoutCandidates(t *testing.T) {
	model := &judgeModel{}
	result, err := New(matchConfig(), model).Match(context.Background(), config.MatchRAG, nil, Input{Request: "q", Index: &index{}})
	if err != nil || len(result.Kept) != 0 || len(model.states()) != 0 {
		t.Fatalf("nothing recalled, nothing judged: %+v %v", result, err)
	}
	if result.Stages[0].Outcome != OutcomeEmpty || result.Stages[1].Outcome != OutcomeEmpty {
		t.Fatalf("stages %+v", result.Stages)
	}
	segments := listed("filled", " ")
	model = &judgeModel{scores: map[string]float64{"filled": 0.7}}
	result, err = New(matchConfig(), model).Match(context.Background(), config.MatchCompress, nil, Input{Request: "q", Candidates: segments})
	if err != nil || len(model.states()) != 1 || !reflect.DeepEqual(ids(result.Kept), []string{"c0", "c1"}) || result.Kept[1].Score != 0 {
		t.Fatalf("a candidate without text scores 0 unasked: %+v %v, %d states", result.Kept, err, len(model.states()))
	}
}

func TestMatchRefusesInputItCannotScore(t *testing.T) {
	matcher := New(matchConfig(), &judgeModel{})
	for name, input := range map[string]Input{
		"empty request":       {Request: " ", Candidates: listed("a")},
		"duplicate IDs":       {Request: "q", Candidates: []Candidate{{ID: "a"}, {ID: "a"}}},
		"missing ID":          {Request: "q", Candidates: []Candidate{{Text: "a"}}},
		"judge over an index": {Request: "q", Index: &index{}},
	} {
		if _, err := matcher.Match(context.Background(), config.MatchCompress, nil, input); err == nil {
			t.Fatalf("%s: expected an error", name)
		}
	}
	duplicate := &index{recalled: []Scored{{Candidate: Candidate{ID: "x"}}, {Candidate: Candidate{ID: "x"}}}}
	if _, err := matcher.Match(context.Background(), config.MatchRAG, nil, Input{Request: "q", Index: duplicate}); err == nil {
		t.Fatal("an index that recalls one ID twice fails the recall")
	}
}

func TestEveryStageIsAMetric(t *testing.T) {
	judged := judgeTotal.WithLabelValues(string(config.MatchRAG), presets.Relevance, OutcomeSkipped)
	before, recalledBefore := testutil.ToFloat64(judged), testutil.ToFloat64(recallTotal.WithLabelValues(string(config.MatchRAG), config.MatchRecallEmbedding, OutcomeOK))
	if _, err := New(matchConfig(), &judgeModel{err: modelservice.ErrUnavailable}).Match(context.Background(), config.MatchRAG, nil, Input{Request: "q", Index: &index{recalled: recalled(0.9)}}); err != nil {
		t.Fatal(err)
	}
	if testutil.ToFloat64(judged) != before+1 || testutil.ToFloat64(recallTotal.WithLabelValues(string(config.MatchRAG), config.MatchRecallEmbedding, OutcomeOK)) != recalledBefore+1 {
		t.Fatal("each stage counts once by feature, method and outcome")
	}
}
