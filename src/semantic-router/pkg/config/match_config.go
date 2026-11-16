package config

import (
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// MatchConfig is the one block every feature that finds similar items
// writes. Recall narrows the candidates by embedding or BM25 similarity;
// judge asks a System One question and keeps the candidates it scores best.
// A block runs the stages it writes, at least one; a feature that writes no
// block runs its default (MatchFeature).
type MatchConfig struct {
	// Recall narrows the candidates by index similarity.
	Recall *MatchRecall `json:"recall,omitempty" yaml:"recall,omitempty"`
	// Judge scores the candidates with a System One question.
	Judge *MatchJudge `json:"judge,omitempty" yaml:"judge,omitempty"`
}

// MatchRecall bounds the candidates recall returns, most similar first.
type MatchRecall struct {
	// Method is embedding (cosine similarity in the store's embedding space,
	// the default) or bm25 (the lexical index).
	Method string `json:"method,omitempty" yaml:"method,omitempty" jsonschema:"enum=embedding,enum=bm25"`
	// TopK is the most candidates recall returns; required when they come
	// from an index.
	TopK int `json:"top_k,omitempty" yaml:"top_k,omitempty"`
	// MinSimilarity is the least similarity a candidate needs: a cosine in
	// [-1, 1] for embedding, a BM25 score (at least 0) for bm25.
	MinSimilarity *float64 `json:"min_similarity,omitempty" yaml:"min_similarity,omitempty"`
}

// MatchJudge asks one System One question about the request and the
// candidates, in one call, and keeps candidates by their scores. A choice
// question chooses among the candidates, which are its options, and scores
// each with its probability. A noul or score question is asked about each
// candidate, and scores it with P(yes) or with the expected level. A judge
// reads at most max_candidates candidates and truncates the rest, never
// refuses them.
type MatchJudge struct {
	// Preset is a canonical question: relevance (does the candidate help
	// answer the request), same_request (does the candidate request ask the
	// same thing) or tool_rank (which candidate tool to call). Write it or a
	// question.
	Preset string `json:"preset,omitempty" yaml:"preset,omitempty" jsonschema:"enum=relevance,enum=same_request,enum=tool_rank"`
	// Question is a custom question about the request and a candidate:
	// choice (without choices: the candidates are the options), noul or
	// score.
	Question *SystemOneQuestion `json:"question,omitempty" yaml:"question,omitempty"`
	// Deployment answers the question; without one the decision model does.
	Deployment string `json:"deployment,omitempty" yaml:"deployment,omitempty"`
	// TopK keeps at most this many of the best-scored candidates.
	TopK int `json:"top_k,omitempty" yaml:"top_k,omitempty"`
	// MinScore is the least score a kept candidate has: a probability for
	// choice and noul, an expected level for score.
	MinScore *float64 `json:"min_score,omitempty" yaml:"min_score,omitempty"`
	// MinRatio keeps the candidates that score at least this fraction of
	// the best one, in (0, 1].
	MinRatio *float64 `json:"min_ratio,omitempty" yaml:"min_ratio,omitempty"`
	// MaxCandidates is the most candidates the judge reads: the first by
	// recall rank, or, without recall, as the feature lists them, most
	// recent first. A choice judge reads at most 255, the options of one
	// question, and does by default; one asked about each candidate reads
	// 64 by default and at most 4096.
	MaxCandidates int `json:"max_candidates,omitempty" yaml:"max_candidates,omitempty"`
	// OnError is what a failed or late judge does: skip (the default) keeps
	// the recall result, or the candidates as they came without recall;
	// fail fails the match.
	OnError string `json:"on_error,omitempty" yaml:"on_error,omitempty" jsonschema:"enum=fail,enum=skip"`
	// TimeoutMs bounds the judge's call, up to 60000; without it a systemone
	// deployment's own timeout_ms applies, else 1000.
	TimeoutMs int `json:"timeout_ms,omitempty" yaml:"timeout_ms,omitempty"`
}

// Recall methods and judge failure policies.
const (
	MatchRecallEmbedding = "embedding"
	MatchRecallBM25      = "bm25"
	MatchOnErrorSkip     = "skip"
	MatchOnErrorFail     = "fail"
)

// DefaultMatchJudgeCandidates is how many candidates a judge asked about
// each candidate reads without max_candidates: one batch entry each.
const DefaultMatchJudgeCandidates = 64

// MaxMatchJudgeCandidates is the most candidates a judge asked about each
// candidate reads: the states one /v1/bundle request to a model runtime
// carries at its default --max-bundle-tasks, 64 tasks of 64 states. More go
// in several bundles at once, and the runtime runs one and refuses the rest.
const MaxMatchJudgeCandidates = 64 * 64

// EffectiveMethod is the recall method.
func (r MatchRecall) EffectiveMethod() string {
	if r.Method == "" {
		return MatchRecallEmbedding
	}
	return r.Method
}

// EffectiveOnError is what a failed judge does.
func (j MatchJudge) EffectiveOnError() string {
	if j.OnError == "" {
		return MatchOnErrorSkip
	}
	return j.OnError
}

// EffectiveTimeout is the judge's call deadline.
func (j MatchJudge) EffectiveTimeout() time.Duration {
	return systemOneTimeout(j.TimeoutMs)
}

// QuestionType is the type of the question the judge asks.
func (j MatchJudge) QuestionType() string {
	if j.Question != nil {
		return j.Question.Type
	}
	preset, _ := presets.Lookup(j.Preset, presets.StageRequest)
	return preset.Type
}

// Chooses reports whether the judge chooses among the candidates in one
// question, rather than asking about each.
func (j MatchJudge) Chooses() bool {
	return j.QuestionType() == SystemOneQuestionChoice
}

// EffectiveMaxCandidates is the most candidates the judge reads.
func (j MatchJudge) EffectiveMaxCandidates() int {
	switch {
	case j.MaxCandidates > 0:
		return j.MaxCandidates
	case j.Chooses():
		return presets.MaxChoices
	}
	return DefaultMatchJudgeCandidates
}

// MatchJudgeDeployment returns the deployment a judge asks: its own, or the
// decision model's.
func (c *RouterConfig) MatchJudgeDeployment(judge MatchJudge) string {
	return c.deploymentOrDecisionModel(judge.Deployment)
}

// MatchFeature is a feature that finds similar items with a match block.
type MatchFeature string

// The features with a match block, by the name their metrics and trace
// records carry.
const (
	MatchResponseCache MatchFeature = "response_cache"
	MatchRAG           MatchFeature = "rag"
	MatchMemory        MatchFeature = "memory"
	MatchCompress      MatchFeature = "compress"
	MatchToolsSelect   MatchFeature = "tools"
)

// matchFeature is a feature's row of the defaults table: how its messages
// name it, the block it runs when it writes none, whether its candidates
// live in an index, which a judge without recall reads only when the
// feature lists them whole (listed), and the recall methods its candidates
// are recalled by.
type matchFeature struct {
	noun          string
	defaults      MatchConfig
	indexed       bool
	listed        bool
	recallMethods []string
}

// matchFeatures is the defaults table. The response cache recalls cached
// requests and serves one that same_request confirms; a RAG source recalls
// widely and lets the relevance judge decide; a memory source recalls;
// compression prefilters the request's own segments by BM25 and judges
// them, and its budget bounds what it keeps; tool selection chooses among
// at most 255 tools.
var matchFeatures = map[MatchFeature]matchFeature{
	MatchResponseCache: {
		noun: "the response cache", indexed: true, recallMethods: []string{MatchRecallEmbedding},
		defaults: MatchConfig{
			Recall: &MatchRecall{Method: MatchRecallEmbedding, TopK: 3, MinSimilarity: matchBound(0.8)},
			Judge:  &MatchJudge{Preset: presets.SameRequest, MinScore: matchBound(0.8)},
		},
	},
	MatchRAG: {
		noun: "a RAG source", indexed: true, listed: true, recallMethods: []string{MatchRecallEmbedding, MatchRecallBM25},
		defaults: MatchConfig{
			Recall: &MatchRecall{Method: MatchRecallEmbedding, TopK: 20, MinSimilarity: matchBound(0.3)},
			Judge:  &MatchJudge{Preset: presets.Relevance, TopK: 6, MinScore: matchBound(presets.NeutralThreshold)},
		},
	},
	MatchMemory: {
		noun: "a memory source", indexed: true, listed: true, recallMethods: []string{MatchRecallEmbedding},
		defaults: MatchConfig{Recall: &MatchRecall{Method: MatchRecallEmbedding, TopK: 5, MinSimilarity: matchBound(0.7)}},
	},
	MatchCompress: {
		noun: "context compression", recallMethods: []string{MatchRecallBM25},
		defaults: MatchConfig{
			Recall: &MatchRecall{Method: MatchRecallBM25},
			Judge:  &MatchJudge{Preset: presets.Relevance},
		},
	},
	MatchToolsSelect: {
		noun: "tool selection", recallMethods: []string{MatchRecallBM25},
		defaults: MatchConfig{Judge: &MatchJudge{Preset: presets.ToolRank, TopK: 3, MinRatio: matchBound(0.1)}},
	},
}

// Default is the block the feature runs when it writes none.
func (f MatchFeature) Default() MatchConfig {
	return matchFeatures[f].defaults.clone()
}

// Effective is the block the feature runs: the one it writes, else its
// default.
func (f MatchFeature) Effective(block *MatchConfig) MatchConfig {
	if block == nil {
		return f.Default()
	}
	return block.clone()
}

func (m MatchConfig) clone() MatchConfig {
	if m.Recall != nil {
		recall := *m.Recall
		recall.MinSimilarity = cloneBound(recall.MinSimilarity)
		m.Recall = &recall
	}
	if m.Judge != nil {
		judge := *m.Judge
		judge.MinScore, judge.MinRatio = cloneBound(judge.MinScore), cloneBound(judge.MinRatio)
		if judge.Question != nil {
			question := *judge.Question
			question.Choices = slices.Clone(question.Choices)
			question.Levels = slices.Clone(question.Levels)
			question.Labels = slices.Clone(question.Labels)
			question.Threshold = cloneBound(question.Threshold)
			judge.Question = &question
		}
		m.Judge = &judge
	}
	return m
}

func matchBound(value float64) *float64 {
	return &value
}

func cloneBound(value *float64) *float64 {
	if value == nil {
		return nil
	}
	return matchBound(*value)
}
