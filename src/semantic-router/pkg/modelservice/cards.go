package modelservice

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// ModelCard is a served model's description from /v1/models: identity,
// surfaces, heads, embedding and rerank exits, limits and placement.
// Preparation checks every task binding against it.
type ModelCard struct {
	ID             string
	Family         string
	Repo           string
	Revision       string
	ModelSHA256    string
	ManifestSHA256 string
	Surfaces       []string
	Heads          []HeadCard
	Embedding      *EmbeddingCard
	Rerank         *RerankCard
	MaxInputTokens int
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
// package declares.
type HeadCard struct {
	Name             string
	Kind             string
	Labels           []string
	Inputs           []string
	DefaultThreshold *float64
	Thresholds       []float64
	Overflow         string
	Window           *Window
	Reduction        string
}

// EmbeddingCard lists the dimensions and layer exits a pooled model serves.
type EmbeddingCard struct {
	Dimensions []int
	Layers     []int
	Modalities []string
	InputTypes []string
	Normalized bool
	Pooling    string
}

// RerankCard lists the pair-scorer exits of a relevance model.
type RerankCard struct {
	Default RerankExit
	Exits   []RerankExit
}

// RerankExit is one pair-scorer exit.
type RerankExit struct {
	Layer     int
	Dimension int
}

// Serves reports whether the model serves a surface (classify, embeddings, ...).
func (c ModelCard) Serves(surface string) bool {
	return slices.Contains(c.Surfaces, surface)
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
		Repo: deref(card.Repo), Revision: deref(card.Revision), ModelSHA256: deref(card.ModelSha256),
		ManifestSHA256: deref(card.ManifestSha256), Profile: deref(card.Profile), Engine: deref(card.Engine),
		Accelerator: deref(card.Accelerator), Device: deref(card.Device), Dtype: deref(card.Dtype),
		Status: deref(card.Status), Reason: deref(card.Reason),
	}
	if card.Limits != nil {
		decoded.MaxInputTokens = deref(card.Limits.MaxInputTokens)
		decoded.MaxInputs = deref(card.Limits.MaxInputs)
	}
	if card.Heads != nil {
		for _, head := range *card.Heads {
			decoded.Heads = append(decoded.Heads, decodeHead(head))
		}
	}
	if card.Embedding != nil {
		decoded.Embedding = &EmbeddingCard{
			Dimensions: append([]int(nil), card.Embedding.Dimensions...),
			Layers:     append([]int(nil), card.Embedding.Layers...),
			Modalities: derefSlice(card.Embedding.Modalities),
			InputTypes: derefSlice(card.Embedding.InputTypes),
			Normalized: deref(card.Embedding.Normalized),
			Pooling:    deref(card.Embedding.Pooling),
		}
	}
	if card.Rerank != nil {
		decoded.Rerank = &RerankCard{Default: RerankExit{Layer: card.Rerank.Default.Layer, Dimension: card.Rerank.Default.Dimension}}
		for _, exit := range card.Rerank.Exits {
			decoded.Rerank.Exits = append(decoded.Rerank.Exits, RerankExit{Layer: exit.Layer, Dimension: exit.Dimension})
		}
	}
	return decoded
}

func decodeHead(head api.HeadCard) HeadCard {
	decoded := HeadCard{
		Name: head.Name, Kind: string(head.Kind), Labels: append([]string(nil), head.Labels...),
		Inputs: derefSlice(head.Inputs), DefaultThreshold: head.DefaultThreshold, Thresholds: derefSlice(head.Thresholds),
		Overflow: deref(head.Overflow), Reduction: deref(head.Reduction),
	}
	if head.Window != nil && head.Window.Tokens != nil {
		decoded.Window = &Window{Tokens: *head.Window.Tokens, Overlap: deref(head.Window.Overlap)}
	}
	return decoded
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
