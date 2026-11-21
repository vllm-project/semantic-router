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

// ErrModelNotPrepared reports a model the generation prepared no provider for:
// none of its consumers asked for it.
var ErrModelNotPrepared = errors.New("was not prepared for this generation")

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
		return nil, fmt.Errorf("%w: embedding deployment %q %w", binding.ErrCapability, model, ErrModelNotPrepared)
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

// Primary names the set's default provider: the deployment
// global.model_catalog.embeddings.default names.
func (s *Set) Primary() string {
	if s == nil {
		return ""
	}
	return s.primary
}

type ModelInfo struct {
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
		info.Modalities = append([]string(nil), info.Modalities...)
		info.Layers = append([]int(nil), info.Layers...)
		info.Dimensions = append([]int(nil), info.Dimensions...)
		infos = append(infos, info)
	}
	sort.Slice(infos, func(i, j int) bool { return infos[i].Name < infos[j].Name })
	return infos
}
