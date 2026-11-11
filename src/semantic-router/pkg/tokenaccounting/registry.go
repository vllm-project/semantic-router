// Package tokenaccounting counts a neutral request's prompt tokens with the
// tokenizer of the model that will receive it, and says where every count
// came from.
//
// A Registry maps model names to tokenizer families. It is built once, off the
// request path, from tokenizers that are already loaded; counting never loads
// or fetches tokenizer assets. A Counter walks every prompt-bearing component
// of an llmprotocol.Request: messages, instructions, reasoning text, tool
// calls, tool results, tool definitions and the output schema. With a
// registered tokenizer the text is counted exactly (SourceExact); media and
// per-item chat framing stay conservative reserves because no text tokenizer
// can count them. Without one, the Counter keeps the provider-neutral estimate,
// adjusted by online calibration when the caller has a calibrated ratio
// (SourceCalibrated), else unchanged (SourceFallback).
package tokenaccounting

import (
	"errors"
	"fmt"
	"strings"
)

// Source records how a token count was produced.
type Source string

const (
	// SourceExact means the model's tokenizer counted the text.
	SourceExact Source = "exact"
	// SourceCalibrated means the provider-neutral estimate with the text
	// rate replaced by an online-calibrated bytes-per-token ratio.
	SourceCalibrated Source = "calibrated"
	// SourceFallback means the provider-neutral estimate, unchanged.
	SourceFallback Source = "fallback"
)

// Tokenizer counts the tokens a model's tokenizer produces for text, without
// special tokens or a chat template. Implementations must be safe for
// concurrent use and must not perform I/O.
type Tokenizer interface {
	CountTokens(text string) int
}

// Family is one tokenizer and the model names that use it.
type Family struct {
	Name      string
	Models    []string
	Tokenizer Tokenizer
}

// Registry resolves a model name to its tokenizer family. It is immutable
// after NewRegistry, so lookups need no locking.
type Registry struct {
	byModel map[string]*Family
}

// NewRegistry validates and indexes families. Model names match exactly
// after trimming spaces; a model may belong to only one family.
func NewRegistry(families ...Family) (*Registry, error) {
	registry := &Registry{byModel: map[string]*Family{}}
	seen := map[string]bool{}
	for index := range families {
		family := families[index]
		family.Name = strings.TrimSpace(family.Name)
		if family.Name == "" {
			return nil, fmt.Errorf("tokenizer family %d has no name", index)
		}
		if seen[family.Name] {
			return nil, fmt.Errorf("tokenizer family %q is defined twice", family.Name)
		}
		seen[family.Name] = true
		if family.Tokenizer == nil {
			return nil, fmt.Errorf("tokenizer family %q has no tokenizer", family.Name)
		}
		if len(family.Models) == 0 {
			return nil, fmt.Errorf("tokenizer family %q lists no models", family.Name)
		}
		entry := &Family{Name: family.Name, Tokenizer: family.Tokenizer}
		for _, model := range family.Models {
			model = strings.TrimSpace(model)
			if model == "" {
				return nil, fmt.Errorf("tokenizer family %q lists an empty model name", family.Name)
			}
			if owner, ok := registry.byModel[model]; ok {
				return nil, fmt.Errorf("model %q is in tokenizer families %q and %q", model, owner.Name, family.Name)
			}
			entry.Models = append(entry.Models, model)
			registry.byModel[model] = entry
		}
	}
	return registry, nil
}

// ErrNoTokenizer reports that no registered family covers a model.
var ErrNoTokenizer = errors.New("no tokenizer registered for model")

// Resolve returns the family for model. A nil Registry resolves nothing.
func (registry *Registry) Resolve(model string) (*Family, error) {
	if registry == nil {
		return nil, ErrNoTokenizer
	}
	family, ok := registry.byModel[strings.TrimSpace(model)]
	if !ok {
		return nil, ErrNoTokenizer
	}
	return family, nil
}
