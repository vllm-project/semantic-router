package modelservice

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// ModelCard is a served model's description from /v1/models: identity,
// surfaces, the question types and presets a decision model answers, heads,
// embedding and rerank exits, limits and placement. Preparation checks every
// task binding against it. MaxScanTokens is the scan budget of a decision
// model that reads a long state part in windows (Vela 2.0), zero for one that
// reads one bounded input.
type ModelCard struct {
	ID             string
	Family         string
	Repo           string
	Revision       string
	ModelSHA256    string
	ManifestSHA256 string
	Surfaces       []string
	QuestionTypes  []string
	Presets        []string
	Heads          []HeadCard
	Embedding      *EmbeddingCard
	Rerank         *RerankCard
	MaxInputTokens int
	MaxScanTokens  int
	MaxInputs      int
	Profile        string
	Engine         string
	Accelerator    string
	Device         string
	Dtype          string
	Ready          bool
	Status         string
	Reason         string
}

// HeadCard is one classify head: its kind (sequence, scores or token), label
// order, accepted inputs and the overflow, window and threshold policy the
// package declares. OperatingPointSHA256 is the digest of the verified policy
// file the head applies, if any.
type HeadCard struct {
	Name                 string
	Kind                 string
	Labels               []string
	Inputs               []string
	DefaultThreshold     *float64
	Thresholds           []float64
	Overflow             string
	Window               *Window
	Reduction            string
	OperatingPointSHA256 string
}

// EmbeddingCard lists the dimensions and layer exits a pooled model serves.
type EmbeddingCard = api.EmbeddingCard

// RerankCard lists the pair-scorer exits of a relevance model.
type RerankCard = api.RerankCard

// RerankExit is one pair-scorer exit.
type RerankExit = api.RerankExit

// Serves reports whether the model serves a surface (classify, embeddings, ...).
func (c ModelCard) Serves(surface string) bool {
	return slices.Contains(c.Surfaces, surface)
}

// Answers reports whether the model answers a question type on /v1/decisions.
// A card that lists no question types answers the System One types (choice,
// noul and score).
func (c ModelCard) Answers(questionType string) bool {
	if !c.Serves("decisions") {
		return false
	}
	if len(c.QuestionTypes) == 0 {
		return questionType == "choice" || questionType == "noul" || questionType == "score"
	}
	return slices.Contains(c.QuestionTypes, questionType)
}

// HasPreset reports whether the model defines the named question.
func (c ModelCard) HasPreset(name string) bool {
	return c.Serves("decisions") && slices.Contains(c.Presets, name)
}

// Head returns the named head, or the primary (first) head when name is empty.
func (c ModelCard) Head(name string) (HeadCard, bool) {
	if name == "" {
		if len(c.Heads) == 0 {
			return HeadCard{}, false
		}
		return c.Heads[0], true
	}
	for _, head := range c.Heads {
		if head.Name == name {
			return head, true
		}
	}
	return HeadCard{}, false
}

// Accepts reports whether the head takes an input form (text, pair, grounded).
// A head that lists no inputs takes text.
func (h HeadCard) Accepts(input string) bool {
	if len(h.Inputs) == 0 {
		return input == "text"
	}
	return slices.Contains(h.Inputs, input)
}

func decodeCard(card api.ModelCard) ModelCard {
	decoded := ModelCard{
		ID: card.Id, Family: card.Family, Surfaces: append([]string(nil), card.Surfaces...), Ready: card.Ready,
		QuestionTypes: derefSlice(card.QuestionTypes), Presets: derefSlice(card.Presets),
		Repo: deref(card.Repo), Revision: deref(card.Revision), ModelSHA256: deref(card.ModelSha256),
		ManifestSHA256: deref(card.ManifestSha256), Profile: deref(card.Profile), Engine: deref(card.Engine),
		Accelerator: deref(card.Accelerator), Device: deref(card.Device), Dtype: deref(card.Dtype),
		Status: string(deref(card.Status)), Reason: deref(card.Reason),
	}
	if card.Limits != nil {
		decoded.MaxInputTokens = deref(card.Limits.MaxInputTokens)
		decoded.MaxScanTokens = deref(card.Limits.MaxScanTokens)
		decoded.MaxInputs = deref(card.Limits.MaxInputs)
	}
	if card.Heads != nil {
		for _, head := range *card.Heads {
			decoded.Heads = append(decoded.Heads, decodeHead(head))
		}
	}
	decoded.Embedding, decoded.Rerank = card.Embedding, card.Rerank
	return decoded
}

// ModelCardFromAPI adapts canonical runtime observations for persistent
// control-plane consumers, including limits, heads, presets and question types.
func ModelCardFromAPI(card api.ModelCard) ModelCard { return decodeCard(card) }

func decodeHead(head api.HeadCard) HeadCard {
	return HeadCard{
		Name: head.Name, Kind: string(head.Kind), Labels: append([]string(nil), head.Labels...),
		Inputs: derefSlice(head.Inputs), DefaultThreshold: head.DefaultThreshold, Thresholds: derefSlice(head.Thresholds),
		Overflow: string(deref(head.Overflow)), Window: head.Window, Reduction: string(deref(head.Reduction)),
		OperatingPointSHA256: deref(head.OperatingPointSha256),
	}
}

func deref[T any](value *T) T {
	var zero T
	if value == nil {
		return zero
	}
	return *value
}

func derefSlice[T any](values *[]T) []T {
	if values == nil {
		return nil
	}
	return append([]T(nil), (*values)...)
}
