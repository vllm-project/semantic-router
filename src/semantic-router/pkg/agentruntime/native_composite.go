package agentruntime

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

// CompositeNativeToolProvider merges narrow domain adapters and enforces the
// exact Router-owned Builder surface. Missing, duplicate, or unexpected tools
// fail registry construction instead of silently degrading a session.
type CompositeNativeToolProvider struct {
	providers []NativeToolProvider
	required  []string
}

func NewCompositeNativeToolProvider(
	providers []NativeToolProvider, required []string,
) (*CompositeNativeToolProvider, error) {
	if len(providers) == 0 || len(required) == 0 {
		return nil, errors.New("agent native Tool providers and required names are incomplete")
	}
	seen := make(map[string]struct{}, len(required))
	canonical := append([]string(nil), required...)
	sort.Strings(canonical)
	for _, name := range canonical {
		if name == "" {
			return nil, errors.New("agent required native Tool name is empty")
		}
		if _, duplicate := seen[name]; duplicate {
			return nil, fmt.Errorf("duplicate required Agent native Tool %q", name)
		}
		seen[name] = struct{}{}
	}
	return &CompositeNativeToolProvider{
		providers: append([]NativeToolProvider(nil), providers...), required: canonical,
	}, nil
}

func (provider *CompositeNativeToolProvider) Current(
	ctx context.Context, namespaceID string,
) ([]agentmanagement.RegisteredTool, error) {
	if provider == nil {
		return nil, agentmanagement.ErrToolUnavailable
	}
	tools := make([]agentmanagement.RegisteredTool, 0, len(provider.required))
	seen := make(map[string]struct{}, len(provider.required))
	for _, source := range provider.providers {
		if source == nil {
			return nil, agentmanagement.ErrToolUnavailable
		}
		current, err := source.Current(ctx, namespaceID)
		if err != nil {
			return nil, err
		}
		for _, tool := range current {
			if _, duplicate := seen[tool.Definition.Name]; duplicate {
				return nil, fmt.Errorf("%w: duplicate Router-native Tool %q", agentmanagement.ErrConflict, tool.Definition.Name)
			}
			seen[tool.Definition.Name] = struct{}{}
			tools = append(tools, tool)
		}
	}
	if len(tools) != len(provider.required) {
		return nil, fmt.Errorf("%w: Router-native Tool set is incomplete", agentmanagement.ErrToolUnavailable)
	}
	for _, required := range provider.required {
		if _, found := seen[required]; !found {
			return nil, fmt.Errorf("%w: required Router-native Tool %q is unavailable", agentmanagement.ErrToolUnavailable, required)
		}
	}
	sort.Slice(tools, func(left, right int) bool {
		return tools[left].Definition.Name < tools[right].Definition.Name
	})
	return tools, nil
}

func (provider *CompositeNativeToolProvider) Resolve(
	ctx context.Context, namespaceID string, definition agentmanagement.ToolDefinition,
) (agentmanagement.ToolHandler, error) {
	if provider == nil {
		return nil, agentmanagement.ErrToolUnavailable
	}
	for _, source := range provider.providers {
		handler, err := source.Resolve(ctx, namespaceID, definition)
		if errors.Is(err, agentmanagement.ErrToolUnavailable) {
			continue
		}
		return handler, err
	}
	return nil, agentmanagement.ErrToolUnavailable
}

var _ NativeToolProvider = (*CompositeNativeToolProvider)(nil)
