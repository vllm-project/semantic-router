package masking

import (
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/maskingregistry"
)

// registryIndexPattern is a canonical int64 index: no leading zeros, at most
// 19 digits. A longer run cannot match, since a template closes its index with
// a character other than a digit on both sides.
const registryIndexPattern = `(0|[1-9][0-9]{0,18})`

// NewRegistryAllocator returns an allocator that takes its indexes from
// identity, in the scope of principal and rootSession, so a value keeps its
// placeholder in every request of that scope.
func NewRegistryAllocator(cfg *config.MaskingPluginConfig, identity *maskingregistry.Identity, principal, rootSession string) (*Allocator, error) {
	if !cfg.RegistryMode() || identity == nil {
		return nil, fmt.Errorf("masking: registry allocator needs an identity namespace")
	}
	scope, err := identity.Scope(principal, rootSession)
	if err != nil {
		return nil, err
	}
	a := NewAllocator(cfg)
	a.identity, a.scope, a.err = identity, scope, nil
	a.pending = map[int]map[int64]string{}
	a.defaults = map[string]bool{}
	for _, shape := range a.shapes {
		if err := a.claim(shape.entityType); err != nil {
			return nil, err
		}
	}
	return a, nil
}

func (a *Allocator) claim(entityType string) error {
	if entityType == "" {
		return nil
	}
	return a.identity.Claim(a.cfg.CanonicalType(entityType), a.cfg.EffectivePlaceholder(entityType))
}

// prepare reads every text and JSON string leaf the walk masks before any
// value is allocated, then asks the registry once per type which of the
// placeholders found there it issued, reserving the others for good. A
// literal placeholder in a later message is thereby never handed to a value
// in an earlier one.
func (a *Allocator) prepare(request *llmprotocol.Request) error {
	a.collecting = true
	_, err := applyRequest(request, a, func(string) ([]Span, error) { return nil, nil })
	a.collecting = false
	if err == nil {
		err = a.settle()
	}
	if err != nil {
		a.err = err
	}
	return err
}

// collect notes the placeholders in text the registry has not answered for.
func (a *Allocator) collect(text string) {
	for _, loc := range occurrences(anyDefaultShape, text) {
		a.defaults[text[loc[0]:loc[1]]] = true
	}
	for i, shape := range a.shapes {
		if shape.entityType != "" {
			a.collectShape(i, text)
		}
	}
}

func (a *Allocator) collectShape(i int, text string) {
	for _, loc := range occurrences(a.shapes[i].pattern, text) {
		found := text[loc[0]:loc[1]]
		if _, known := a.issued[found]; known {
			continue
		}
		index, err := strconv.ParseInt(text[loc[2]:loc[3]], 10, 64)
		if err != nil {
			// Past int64: never issued, and never allocated either.
			continue
		}
		if a.pending[i] == nil {
			a.pending[i] = map[int64]string{}
		}
		a.pending[i][index] = found
	}
}

// settle answers the collected placeholders, in shape order.
func (a *Allocator) settle() error {
	for i, shape := range a.shapes {
		pending := a.pending[i]
		if len(pending) == 0 {
			continue
		}
		indices := slices.Sorted(maps.Keys(pending))
		issued, err := a.identity.Store.Prepare(a.scope, a.cfg.CanonicalType(shape.entityType), indices)
		if err != nil {
			return err
		}
		for _, index := range indices {
			a.issued[pending[index]] = issued[index]
		}
		delete(a.pending, i)
	}
	return nil
}

// registryPlaceholder is Placeholder in registry mode. A label the detector
// answers outside the pii preset gets its shape here, and the default-shaped
// text the request held is answered for it before its first index.
func (a *Allocator) registryPlaceholder(entityType, value string) string {
	key := entityType + "\x00" + value
	if existing, ok := a.assigned[key]; ok {
		return existing
	}
	if !a.shaped[entityType] {
		a.addShape(entityType)
		if a.err = a.claim(entityType); a.err != nil {
			return ""
		}
		last := len(a.shapes) - 1
		for found := range a.defaults {
			a.collectShape(last, found)
		}
		if a.err = a.settle(); a.err != nil {
			return ""
		}
	}
	typ := a.cfg.CanonicalType(entityType)
	index, err := a.identity.Store.Allocate(a.scope, typ, a.identity.Digest(a.scope, typ, value))
	if err != nil {
		a.err = err
		return ""
	}
	placeholder := strings.ReplaceAll(a.cfg.EffectivePlaceholder(entityType), config.MaskingPlaceholderIndexToken, strconv.FormatInt(index, 10))
	a.assigned[key] = placeholder
	a.issued[placeholder] = true
	return placeholder
}

func registryTemplatePattern(template string) string {
	return strings.ReplaceAll(regexp.QuoteMeta(template), regexp.QuoteMeta(config.MaskingPlaceholderIndexToken), registryIndexPattern)
}
