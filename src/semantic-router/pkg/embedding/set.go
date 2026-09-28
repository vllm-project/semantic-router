package embedding

import (
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"sort"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// Set is one generation's immutable snapshot of prepared embedding providers.
// Views select dimensions/layers without loading or discovering models in requests.
// Its owner closes it only after the generation's users drain.
type Set struct {
	providers map[string]Provider
	primary   string
	closers   []io.Closer
	once      sync.Once
	closeErr  error
}

func NewSet(providers map[string]Provider, primary string, closers ...io.Closer) *Set {
	copied := make(map[string]Provider, len(providers))
	for name, provider := range providers {
		copied[strings.ToLower(name)] = provider
	}
	return &Set{providers: copied, primary: strings.ToLower(primary), closers: append([]io.Closer(nil), closers...)}
}

func (s *Set) Get(model string, dimension, layer int) (Provider, error) {
	if s == nil {
		return nil, fmt.Errorf("embedding set was not prepared")
	}
	model = strings.ToLower(strings.TrimSpace(model))
	if model == "" {
		model = s.primary
	}
	provider, ok := s.providers[model]
	if !ok {
		return nil, fmt.Errorf("%w: embedding model %q was not prepared for this generation", binding.ErrCapability, model)
	}
	if dimension < 0 || dimension > math.MaxInt32 || layer < 0 || layer > math.MaxInt32 {
		return nil, fmt.Errorf("%w: dimension and layer must fit nonnegative int32", binding.ErrCapability)
	}
	if described, ok := provider.(Described); ok {
		info := described.EmbeddingInfo()
		if dimension > 0 && len(info.Dimensions) > 0 && !slices.Contains(info.Dimensions, dimension) {
			return nil, fmt.Errorf("%w: model %q supports dimensions %v", binding.ErrCapability, model, info.Dimensions)
		}
		if layer > 0 && !slices.Contains(info.Layers, layer) {
			return nil, fmt.Errorf("%w: model %q supports layer exits %v", binding.ErrCapability, model, info.Layers)
		}
	}
	return WithOptions(provider, Options{Dimension: dimension, Layer: layer}), nil
}
func (s *Set) Default() (Provider, error) { return s.Get("", 0, 0) }
func (s *Set) Close() error {
	if s == nil {
		return nil
	}
	s.once.Do(func() {
		for i := len(s.closers) - 1; i >= 0; i-- {
			s.closeErr = errors.Join(s.closeErr, s.closers[i].Close())
		}
	})
	return s.closeErr
}

// Ready reports whether any configured consumer has a prepared provider.
func (s *Set) Ready() bool { return s != nil && len(s.providers) > 0 }

func (s *Set) Has(model string) bool {
	if s == nil {
		return false
	}
	if model == "" {
		model = s.primary
	}
	_, ok := s.providers[strings.ToLower(model)]
	return ok
}

// Select retains the existing embedding endpoint's priority policy, using only
// providers prepared in this snapshot. It never loads a requested model.
func (s *Set) Select(text string, quality, latency float32, dimension int) (string, error) {
	words := len(strings.Fields(text))
	// Input budgets belong to the selected deployment's tokenizer and overflow
	// policy; whitespace counts here only guide the existing model preference.
	preferred := "qwen3"
	if (words <= 512 && quality <= 0.7 && latency > 0.7) || (words > 512 && words <= 2048) || (dimension > 0 && dimension < 768 && latency > 0.5) {
		preferred = "gemma"
	}
	order := []string{preferred, "mmbert", "gemma", "qwen3"}
	for _, model := range order {
		if s.Has(model) {
			return model, nil
		}
	}
	if s.Has("") {
		return s.primary, nil
	}
	return "", fmt.Errorf("no prepared embedding model is available")
}

// SelectModalities chooses one already prepared representation for an entire
// media batch. Prefer the scope's primary model, then deterministic alternatives;
// aliases never imply an encoder capability or trigger model preparation.
func (s *Set) SelectModalities(modalities []string, dimension, layer int) (string, error) {
	if s == nil {
		return "", fmt.Errorf("%w: embedding set was not prepared", binding.ErrCapability)
	}
	models := s.Models()
	sort.SliceStable(models, func(i, j int) bool {
		return models[i].Name == s.primary && models[j].Name != s.primary
	})
	for _, model := range models {
		compatible := true
		for _, modality := range modalities {
			if !slices.Contains(model.Modalities, modality) || (modality == "audio" && model.Audio == nil) {
				compatible = false
				break
			}
		}
		if !compatible {
			continue
		}
		if _, err := s.Get(model.Name, dimension, layer); err == nil {
			return model.Name, nil
		}
	}
	return "", fmt.Errorf("%w: no prepared model supports modalities %v with dimension %d and layer %d", binding.ErrCapability, modalities, dimension, layer)
}

type ModelInfo struct {
	Audio         *binding.AudioCapability
	Name          string
	Artifact      string
	Backend       string
	Dimension     int
	Dimensions    []int
	MaxTokens     int
	Pooling       string
	Normalization string
	Modalities    []string
	Layers        []int
}
type Described interface{ EmbeddingInfo() ModelInfo }

func (s *Set) Models() []ModelInfo {
	if s == nil {
		return nil
	}
	var infos []ModelInfo
	for name, provider := range s.providers {
		info := ModelInfo{Name: name, Backend: provider.Backend(), Dimension: provider.Dimension()}
		if p, ok := provider.(Described); ok {
			info = p.EmbeddingInfo()
			info.Name = name
		}
		if info.Audio != nil {
			a := *info.Audio
			a.SampleRates = append([]int(nil), a.SampleRates...)
			info.Audio = &a
		}
		info.Modalities = append([]string(nil), info.Modalities...)
		info.Layers = append([]int(nil), info.Layers...)
		info.Dimensions = append([]int(nil), info.Dimensions...)
		infos = append(infos, info)
	}
	sort.Slice(infos, func(i, j int) bool { return infos[i].Name < infos[j].Name })
	return infos
}
