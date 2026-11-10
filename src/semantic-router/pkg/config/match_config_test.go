package config

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

func matchConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		DefaultDecisionDeployment: {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/vela3-fast"},
		"lite":                    {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/vela3-lite"},
		"chooser": {
			Provider: SystemOneProvider, Endpoint: "https://api.example.com/v1/systemone", Model: "vllm-sr/d3-ultra",
			Capabilities: &DeploymentCapabilities{QuestionTypes: []string{SystemOneQuestionChoice}},
		},
		"bert": {Provider: OpenAIEmbeddingsProvider, Endpoint: "http://embedder:8000/v1", Model: "bert"},
	}
	return cfg
}

// The defaults table: the cache recalls and confirms with same_request, RAG
// recalls and judges relevance, memory recalls, compression prefilters by
// BM25 and judges relevance, and tool selection ranks tools. Every default
// is a valid block of its own feature.
func TestMatchDefaultsTable(t *testing.T) {
	stages := func(m MatchConfig) string {
		var written []string
		if m.Recall != nil {
			written = append(written, "recall:"+m.Recall.EffectiveMethod())
		}
		if m.Judge != nil {
			written = append(written, "judge:"+m.Judge.Preset)
		}
		return strings.Join(written, " ")
	}
	want := map[MatchFeature]string{
		MatchResponseCache: "recall:embedding judge:same_request",
		MatchRAG:           "recall:embedding judge:relevance",
		MatchMemory:        "recall:embedding",
		MatchCompress:      "recall:bm25 judge:relevance",
		MatchToolsSelect:   "judge:tool_rank",
	}
	if len(matchFeatures) != len(want) {
		t.Fatalf("the table has %d rows", len(matchFeatures))
	}
	for feature, row := range matchFeatures {
		block := feature.Default()
		if got := stages(block); got != want[feature] {
			t.Fatalf("%s default = %q, want %q", feature, got, want[feature])
		}
		if err := validateMatch(matchConfig(), string(feature)+".match", &block, feature); err != nil {
			t.Fatalf("%s default: %v", feature, err)
		}
		if row.indexed && block.Recall == nil {
			t.Fatalf("%s: an indexed feature recalls by default", feature)
		}
	}
	if rag := MatchRAG.Default(); rag.Recall.TopK != 20 || *rag.Recall.MinSimilarity != 0.3 || rag.Judge.TopK != 6 || *rag.Judge.MinScore != presets.NeutralThreshold {
		t.Fatalf("rag default = %+v %+v", rag.Recall, rag.Judge)
	}
	if cache := MatchResponseCache.Default(); cache.Recall.TopK != 3 || *cache.Recall.MinSimilarity != 0.8 || cache.Judge.Preset != presets.SameRequest || *cache.Judge.MinScore != 0.8 {
		t.Fatalf("cache default = %+v %+v", cache.Recall, cache.Judge)
	}
	if compress := MatchCompress.Default(); compress.Recall.TopK != 0 || compress.Recall.MinSimilarity != nil || compress.Judge.EffectiveMaxCandidates() != 64 {
		t.Fatalf("compress default = %+v %+v", compress.Recall, compress.Judge)
	}
}

// A judge reads 255 candidates when it chooses among them, 64 when it is
// asked about each, unless max_candidates says otherwise.
func TestMatchJudgeMaxCandidates(t *testing.T) {
	cases := map[string]struct {
		judge MatchJudge
		want  int
	}{
		"choice":           {MatchJudge{Preset: presets.ToolRank}, presets.MaxChoices},
		"per candidate":    {MatchJudge{Preset: presets.Relevance}, DefaultMatchJudgeCandidates},
		"custom choice":    {MatchJudge{Question: &SystemOneQuestion{Type: SystemOneQuestionChoice}}, presets.MaxChoices},
		"written":          {MatchJudge{Preset: presets.SameRequest, MaxCandidates: 8}, 8},
		"written above 64": {MatchJudge{Preset: presets.Relevance, MaxCandidates: 200}, 200},
	}
	for name, test := range cases {
		if got := test.judge.EffectiveMaxCandidates(); got != test.want {
			t.Fatalf("%s: %d, want %d", name, got, test.want)
		}
	}
}

// A written block replaces the default as a whole, and neither the default
// nor the written block is shared with the block a feature runs.
func TestMatchEffectiveBlock(t *testing.T) {
	block := MatchRAG.Effective(nil)
	*block.Recall.MinSimilarity, block.Judge.TopK = 0.9, 1
	if fresh := MatchRAG.Default(); *fresh.Recall.MinSimilarity != 0.3 || fresh.Judge.TopK != 6 {
		t.Fatalf("a feature's block changed its default: %+v %+v", fresh.Recall, fresh.Judge)
	}
	written := &MatchConfig{Recall: &MatchRecall{TopK: 5, MinSimilarity: matchBound(0.7)}}
	effective := MatchRAG.Effective(written)
	if effective.Judge != nil || !reflect.DeepEqual(effective, *written) {
		t.Fatalf("a written block runs as written: %+v", effective)
	}
	*effective.Recall.MinSimilarity = 0.1
	if *written.Recall.MinSimilarity != 0.7 {
		t.Fatal("the block a feature runs shares the written one")
	}
	judged := MatchToolsSelect.Effective(&MatchConfig{Judge: &MatchJudge{Question: &SystemOneQuestion{Type: SystemOneQuestionChoice, Instructions: "Which tool?"}}})
	if judged.Judge.EffectiveOnError() != MatchOnErrorSkip || judged.Judge.EffectiveTimeout().Milliseconds() != DefaultSystemOneTimeoutMs || !judged.Judge.Chooses() {
		t.Fatalf("judge defaults = %+v", judged.Judge)
	}
	if relevance := (MatchJudge{Preset: presets.Relevance}); relevance.Chooses() || relevance.QuestionType() != SystemOneQuestionNoul {
		t.Fatal("relevance is asked about each candidate")
	}
	if cfg := matchConfig(); cfg.MatchJudgeDeployment(MatchJudge{}) != DefaultDecisionDeployment || cfg.MatchJudgeDeployment(MatchJudge{Deployment: "lite"}) != "lite" {
		t.Fatal("a judge asks its deployment, else the decision model")
	}
}

func TestMatchEmptyBlockFailsWithItsCode(t *testing.T) {
	err := validateMatch(matchConfig(), "recipes[default].decisions[faq].plugins[0].configuration.semantic.match", &MatchConfig{}, MatchResponseCache)
	var located *LocatedError
	if !errors.As(err, &located) || located.Code != CodeMatchEmpty {
		t.Fatalf("err = %v", err)
	}
	if want := "recipes[default].decisions[faq].plugins[0].configuration.semantic.match: match_empty: a match block runs recall, judge or both; write at least one"; err.Error() != want {
		t.Fatalf("err = %q\nwant %q", err, want)
	}
	if err := validateMatch(matchConfig(), "x", nil, MatchResponseCache); err != nil {
		t.Fatalf("a feature without a block runs its default: %v", err)
	}
}

func TestMatchBlockValidation(t *testing.T) {
	recall := func(method string, topK int, min *float64) *MatchRecall {
		return &MatchRecall{Method: method, TopK: topK, MinSimilarity: min}
	}
	judge := func(edit func(*MatchJudge)) *MatchJudge {
		j := &MatchJudge{Preset: presets.Relevance}
		edit(j)
		return j
	}
	question := func(kind string, edit func(*SystemOneQuestion)) func(*MatchJudge) {
		return func(j *MatchJudge) {
			j.Preset, j.Question = "", &SystemOneQuestion{Type: kind, Instructions: "Does it fit?"}
			edit(j.Question)
		}
	}
	none := func(*SystemOneQuestion) {}
	valid := map[string]struct {
		feature MatchFeature
		block   MatchConfig
	}{
		"cache recall and same_request": {MatchResponseCache, MatchConfig{Recall: recall("", 3, matchBound(0.8)), Judge: judge(func(j *MatchJudge) { j.Preset, j.MinScore = presets.SameRequest, matchBound(0.9) })}},
		"compress bm25 bound":           {MatchCompress, MatchConfig{Recall: recall(MatchRecallBM25, 10, matchBound(2.5))}},
		"rag negative cosine bound":     {MatchRAG, MatchConfig{Recall: recall("", 10, matchBound(-0.2))}},
		"compress bm25 without top_k":   {MatchCompress, MatchConfig{Recall: recall(MatchRecallBM25, 0, nil), Judge: judge(func(*MatchJudge) {})}},
		"compress custom noul":          {MatchCompress, MatchConfig{Judge: judge(question(SystemOneQuestionNoul, none))}},
		"compress custom score bound": {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) {
			question(SystemOneQuestionScore, func(q *SystemOneQuestion) { q.Levels = []string{"no", "some", "all"} })(j)
			j.MinScore, j.Deployment = matchBound(2), "lite"
		})}},
		"tools custom choice on chooser": {MatchToolsSelect, MatchConfig{Judge: judge(func(j *MatchJudge) {
			question(SystemOneQuestionChoice, none)(j)
			j.Deployment, j.TopK, j.MinRatio, j.OnError, j.TimeoutMs = "chooser", 2, matchBound(1), MatchOnErrorFail, 2000
		})}},
		"tools on another decision model": {MatchToolsSelect, MatchConfig{Judge: judge(func(j *MatchJudge) { j.Preset, j.Deployment = presets.ToolRank, "lite" })}},
		"tools narrowed by bm25":          {MatchToolsSelect, MatchConfig{Recall: recall(MatchRecallBM25, 300, nil), Judge: judge(func(j *MatchJudge) { j.Preset = presets.ToolRank })}},
		"rag judge only":                  {MatchRAG, MatchConfig{Judge: judge(func(j *MatchJudge) { j.MaxCandidates = 32 })}},
		"rag bm25":                        {MatchRAG, MatchConfig{Recall: recall(MatchRecallBM25, 10, matchBound(1.5)), Judge: judge(func(*MatchJudge) {})}},
		"memory judge only":               {MatchMemory, MatchConfig{Judge: judge(func(*MatchJudge) {})}},
		"choice over more than it reads":  {MatchRAG, MatchConfig{Recall: recall("", 300, nil), Judge: judge(func(j *MatchJudge) { j.Preset = presets.ToolRank })}},
		"choice reading 255":              {MatchToolsSelect, MatchConfig{Judge: judge(func(j *MatchJudge) { j.Preset, j.MaxCandidates = presets.ToolRank, 255 })}},
		"per candidate reading 500":       {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) { j.MaxCandidates = 500 })}},
		"per candidate reading a bundle":  {MatchRAG, MatchConfig{Judge: judge(func(j *MatchJudge) { j.MaxCandidates = 4096 })}},
	}
	for name, test := range valid {
		if err := validateMatch(matchConfig(), "m", &test.block, test.feature); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	invalid := map[string]struct {
		feature MatchFeature
		block   MatchConfig
		want    string
	}{
		"judge only on an index":    {MatchResponseCache, MatchConfig{Judge: judge(func(j *MatchJudge) { j.Preset = presets.SameRequest })}, "m: the response cache recalls its candidates from an index, which a judge cannot read whole; add recall"},
		"embedding over tools":      {MatchToolsSelect, MatchConfig{Recall: recall("", 5, nil), Judge: judge(func(j *MatchJudge) { j.Preset = presets.ToolRank })}, "m.recall.method: tool selection recalls with bm25 only"},
		"embedding over segments":   {MatchCompress, MatchConfig{Recall: recall("", 5, nil)}, "m.recall.method: context compression recalls with bm25 only"},
		"bm25 over cached requests": {MatchResponseCache, MatchConfig{Recall: recall(MatchRecallBM25, 3, nil)}, "m.recall.method: the response cache recalls with embedding only"},
		"unknown method":            {MatchRAG, MatchConfig{Recall: recall("dense", 3, nil)}, "m.recall.method must be embedding or bm25"},
		"index without top_k":       {MatchMemory, MatchConfig{Recall: recall("", 0, nil)}, "m.recall.top_k must be at least 1"},
		"negative top_k":            {MatchCompress, MatchConfig{Recall: recall(MatchRecallBM25, -1, nil)}, "m.recall.top_k must be at least 1"},
		"cosine above one":          {MatchRAG, MatchConfig{Recall: recall("", 5, matchBound(1.5))}, "m.recall.min_similarity must be a cosine similarity within [-1, 1]"},
		"negative bm25 bound":       {MatchCompress, MatchConfig{Recall: recall(MatchRecallBM25, 5, matchBound(-1))}, "m.recall.min_similarity must be a BM25 score of at least 0"},
		"bm25 over memories":        {MatchMemory, MatchConfig{Recall: recall(MatchRecallBM25, 5, nil)}, "m.recall.method: a memory source recalls with embedding only"},
		"preset and question":       {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) { j.Question = &SystemOneQuestion{Type: SystemOneQuestionNoul, Instructions: "x"} })}, "m.judge: a judge asks a preset or a question, not both"},
		"no question":               {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) { j.Preset = "" })}, "m.judge: preset must be relevance, same_request, tool_rank, or write a question"},
		"signal preset":             {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) { j.Preset = presets.Domain })}, "m.judge: preset must be"},
		"set question":              {MatchCompress, MatchConfig{Judge: judge(question(SystemOneQuestionSet, none))}, "m.judge.question: type must be choice, noul or score"},
		"choice with choices": {
			MatchToolsSelect,
			MatchConfig{Judge: judge(question(SystemOneQuestionChoice, func(q *SystemOneQuestion) { q.Choices = []SystemOneOption{{Key: "a"}, {Key: "b"}} }))},
			"m.judge.question: a choice judge chooses among the candidates, which are its options",
		},
		"score without levels": {MatchCompress, MatchConfig{Judge: judge(question(SystemOneQuestionScore, none))}, "m.judge.question: a score question needs 2..10 levels"},
		"undeclared deployment": {
			MatchCompress,
			MatchConfig{Judge: judge(func(j *MatchJudge) { j.Deployment = "missing" })},
			`m.judge: deployment "missing" is not declared in global.model_catalog.deployments`,
		},
		"deployment that is no decision model": {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) { j.Deployment = "bert" })}, `m.judge: deployment "bert" must use provider model_runtime or systemone`},
		"untrimmed deployment":                 {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) { j.Deployment = " lite" })}, "m.judge.deployment must name a model_runtime or systemone deployment"},
		"type the deployment does not answer": {
			MatchCompress,
			MatchConfig{Judge: judge(func(j *MatchJudge) { j.Deployment = "chooser" })},
			`m.judge: deployment "chooser" declares question types choice, so it does not answer a noul question`,
		},
		"choice reading more than one question takes": {
			MatchToolsSelect,
			MatchConfig{Judge: judge(func(j *MatchJudge) { j.Preset, j.MaxCandidates = presets.ToolRank, 256 })},
			"m.judge.max_candidates must be within [1, 255]: a choice judge's candidates are the options of one question",
		},
		"per candidate reading more than a bundle carries": {
			MatchRAG,
			MatchConfig{Judge: judge(func(j *MatchJudge) { j.MaxCandidates = 4097 })},
			"m.judge.max_candidates must be within [1, 4096]: a judge asked about each candidate sends them in one model runtime bundle",
		},
		"negative max_candidates": {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) { j.MaxCandidates = -1 })}, "m.judge.max_candidates must be at least 1 when set"},
		"negative judge top_k":    {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) { j.TopK = -2 })}, "m.judge.top_k must be at least 1 when set"},
		"probability above one":   {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) { j.MinScore = matchBound(1.2) })}, "m.judge.min_score must be within [0, 1]"},
		"level above the top": {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) {
			question(SystemOneQuestionScore, func(q *SystemOneQuestion) { q.Levels = []string{"no", "yes"} })(j)
			j.MinScore, j.Deployment = matchBound(1.5), "lite"
		})}, "m.judge.min_score must be within [0, 1]"},
		"zero ratio":       {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) { j.MinRatio = matchBound(0) })}, "m.judge.min_ratio must be within (0, 1]"},
		"ratio above one":  {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) { j.MinRatio = matchBound(1.1) })}, "m.judge.min_ratio must be within (0, 1]"},
		"unknown on_error": {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) { j.OnError = "ignore" })}, "m.judge.on_error must be skip or fail"},
		"long timeout":     {MatchCompress, MatchConfig{Judge: judge(func(j *MatchJudge) { j.TimeoutMs = MaxSystemOneTimeoutMs + 1 })}, "m.judge.timeout_ms must be within [1, 60000] when set"},
	}
	for name, test := range invalid {
		err := validateMatch(matchConfig(), "m", &test.block, test.feature)
		if err == nil || !strings.HasPrefix(err.Error(), test.want) {
			t.Fatalf("%s: err = %v, want %q", name, err, test.want)
		}
	}
}
