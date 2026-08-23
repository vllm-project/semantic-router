package agentruntime

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

func TestCompositeNativeToolProviderRequiresExactBuilderSurface(t *testing.T) {
	required := agentmanagement.BuiltinBuilderToolNames()
	left := newStaticNativeTools(required[:6])
	right := newStaticNativeTools(required[6:])
	provider, testCompositeNativeToolProviderRequiresExactBuilderSurfaceErr := NewCompositeNativeToolProvider(
		[]NativeToolProvider{left, right}, required,
	)
	if testCompositeNativeToolProviderRequiresExactBuilderSurfaceErr != nil {
		t.Fatal(testCompositeNativeToolProviderRequiresExactBuilderSurfaceErr)
	}
	current, testCompositeNativeToolProviderRequiresExactBuilderSurfaceErr := provider.Current(context.Background(), "namespace")
	if testCompositeNativeToolProviderRequiresExactBuilderSurfaceErr != nil {
		t.Fatal(testCompositeNativeToolProviderRequiresExactBuilderSurfaceErr)
	}
	got := make([]string, len(current))
	for index, registration := range current {
		got[index] = registration.Definition.Name
	}
	want := append([]string(nil), required...)
	sortStrings(want)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("native Tool set = %v, want %v", got, want)
	}

	missing, testCompositeNativeToolProviderRequiresExactBuilderSurfaceErr := NewCompositeNativeToolProvider(
		[]NativeToolProvider{left}, required,
	)
	if testCompositeNativeToolProviderRequiresExactBuilderSurfaceErr != nil {
		t.Fatal(testCompositeNativeToolProviderRequiresExactBuilderSurfaceErr)
	}
	if _, err := missing.Current(context.Background(), "namespace"); !errors.Is(err, agentmanagement.ErrToolUnavailable) {
		t.Fatalf("missing native Tool error = %v", err)
	}

	duplicate, testCompositeNativeToolProviderRequiresExactBuilderSurfaceErr := NewCompositeNativeToolProvider(
		[]NativeToolProvider{left, left, right}, required,
	)
	if testCompositeNativeToolProviderRequiresExactBuilderSurfaceErr != nil {
		t.Fatal(testCompositeNativeToolProviderRequiresExactBuilderSurfaceErr)
	}
	if _, err := duplicate.Current(context.Background(), "namespace"); !errors.Is(err, agentmanagement.ErrConflict) {
		t.Fatalf("duplicate native Tool error = %v", err)
	}
}

type staticNativeTools []agentmanagement.RegisteredTool

func (tools staticNativeTools) Current(
	context.Context, string,
) ([]agentmanagement.RegisteredTool, error) {
	return append([]agentmanagement.RegisteredTool(nil), tools...), nil
}

func (tools staticNativeTools) Resolve(
	_ context.Context, _ string, definition agentmanagement.ToolDefinition,
) (agentmanagement.ToolHandler, error) {
	for _, registration := range tools {
		if registration.Definition.Name == definition.Name {
			return registration.Handler, nil
		}
	}
	return nil, agentmanagement.ErrToolUnavailable
}

func newStaticNativeTools(names []string) staticNativeTools {
	result := make(staticNativeTools, 0, len(names))
	for _, name := range names {
		result = append(result, agentmanagement.RegisteredTool{
			Definition: agentmanagement.ToolDefinition{Name: name},
			Handler: agentmanagement.ToolHandlerFunc(func(
				context.Context, agentmanagement.ToolInvocationContext, json.RawMessage,
			) (agentmanagement.ToolResult, error) {
				return agentmanagement.ToolResult{}, nil
			}),
		})
	}
	return result
}

func sortStrings(values []string) {
	for index := 1; index < len(values); index++ {
		for cursor := index; cursor > 0 && values[cursor] < values[cursor-1]; cursor-- {
			values[cursor], values[cursor-1] = values[cursor-1], values[cursor]
		}
	}
}
