package agentmanagement

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type toolPageRegistrySource struct {
	current *ToolRegistry
}

func (source *toolPageRegistrySource) Current(context.Context, string) (*ToolRegistry, error) {
	return source.current, nil
}

func (source *toolPageRegistrySource) Load(context.Context, string, string) (*ToolRegistry, error) {
	return source.current, nil
}

func TestListToolsUsesRevisionBoundSearchKeyset(t *testing.T) {
	service, source := newToolPageService(t, "router.alpha.read", "router.beta.read", "router.gamma.read")
	namespaceID := uuid.NewString()

	first, revision, testListToolsUsesRevisionBoundSearchKeysetErr := service.ListTools(context.Background(), namespaceID, ToolPageRequest{PageSize: 2})
	if testListToolsUsesRevisionBoundSearchKeysetErr != nil {
		t.Fatal(testListToolsUsesRevisionBoundSearchKeysetErr)
	}
	if len(first.Items) != 2 || first.Items[0].Name != "router.alpha.read" ||
		first.Items[1].Name != "router.beta.read" || !first.HasMore || first.NextCursor == "" {
		t.Fatalf("first page = %+v", first)
	}
	second, secondRevision, testListToolsUsesRevisionBoundSearchKeysetErr := service.ListTools(context.Background(), namespaceID, ToolPageRequest{
		PageSize: 2, Cursor: first.NextCursor,
	})
	if testListToolsUsesRevisionBoundSearchKeysetErr != nil {
		t.Fatal(testListToolsUsesRevisionBoundSearchKeysetErr)
	}
	if secondRevision != revision || len(second.Items) != 1 ||
		second.Items[0].Name != "router.gamma.read" || second.HasMore || second.NextCursor != "" {
		t.Fatalf("second page = %+v, revision = %q", second, secondRevision)
	}

	search, _, testListToolsUsesRevisionBoundSearchKeysetErr := service.ListTools(context.Background(), namespaceID, ToolPageRequest{
		PageSize: 1, Search: "  ROUTER.BETA  ",
	})
	if testListToolsUsesRevisionBoundSearchKeysetErr != nil || len(search.Items) != 1 || search.Items[0].Name != "router.beta.read" {
		t.Fatalf("search page = %+v, error = %v", search, testListToolsUsesRevisionBoundSearchKeysetErr)
	}

	if _, _, err := service.ListTools(context.Background(), namespaceID, ToolPageRequest{
		PageSize: 2, Cursor: first.NextCursor, Search: "router.beta",
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("search-mismatched cursor error = %v", err)
	}
	if _, _, err := service.ListTools(context.Background(), uuid.NewString(), ToolPageRequest{
		PageSize: 2, Cursor: first.NextCursor,
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("namespace-mismatched cursor error = %v", err)
	}
	replacement := "x"
	if strings.HasSuffix(first.NextCursor, replacement) {
		replacement = "y"
	}
	tampered := first.NextCursor[:len(first.NextCursor)-1] + replacement
	if _, _, err := service.ListTools(context.Background(), namespaceID, ToolPageRequest{
		PageSize: 2, Cursor: tampered,
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("tampered cursor error = %v", err)
	}
	description, _, testListToolsUsesRevisionBoundSearchKeysetErr := service.ListTools(context.Background(), namespaceID, ToolPageRequest{
		PageSize: 2, Search: "catalog entry for router.gamma",
	})
	if testListToolsUsesRevisionBoundSearchKeysetErr != nil || len(description.Items) != 1 || description.Items[0].Name != "router.gamma.read" {
		t.Fatalf("description search page = %+v, error = %v", description, testListToolsUsesRevisionBoundSearchKeysetErr)
	}
	updated, testListToolsUsesRevisionBoundSearchKeysetErr := testToolRegistry("router.alpha.read", "router.beta.read", "router.delta.read", "router.gamma.read")
	if testListToolsUsesRevisionBoundSearchKeysetErr != nil {
		t.Fatal(testListToolsUsesRevisionBoundSearchKeysetErr)
	}
	source.current = updated
	if _, _, err := service.ListTools(context.Background(), namespaceID, ToolPageRequest{
		PageSize: 2, Cursor: first.NextCursor,
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale registry cursor error = %v", err)
	}
}

func TestListToolsRejectsUnboundedSearchAndInvalidPage(t *testing.T) {
	service, _ := newToolPageService(t, "router.alpha.read")
	namespaceID := uuid.NewString()
	for _, request := range []ToolPageRequest{
		{PageSize: 0},
		{PageSize: 201},
		{PageSize: 10, Search: strings.Repeat("a", 201)},
		{PageSize: 10, Search: "bad\nsearch"},
	} {
		if _, _, err := service.ListTools(context.Background(), namespaceID, request); !errors.Is(err, ErrInvalid) {
			t.Fatalf("ListTools(%+v) error = %v", request, err)
		}
	}
}

func newToolPageService(t *testing.T, names ...string) (*Service, *toolPageRegistrySource) {
	t.Helper()
	registry, err := testToolRegistry(names...)
	if err != nil {
		t.Fatal(err)
	}
	codec, err := newSignedCodec(securitykeyring.Symmetric{
		ActiveVersion: "cursor-v1",
		Keys:          map[string][]byte{"cursor-v1": []byte("0123456789abcdef0123456789abcdef")},
	})
	if err != nil {
		t.Fatal(err)
	}
	source := &toolPageRegistrySource{current: registry}
	service := &Service{registries: source, codec: codec}
	t.Cleanup(service.Close)
	return service, source
}

func testToolRegistry(names ...string) (*ToolRegistry, error) {
	tools := make([]RegisteredTool, 0, len(names))
	for _, name := range names {
		tools = append(tools, RegisteredTool{
			Definition: ToolDefinition{
				Name: name, Description: "Catalog entry for " + name,
				InputSchema:         json.RawMessage(`{"type":"object","additionalProperties":false}`),
				OutputSchema:        json.RawMessage(`{"type":"object","additionalProperties":false}`),
				RequiredPermissions: []accesscontrol.Permission{accesscontrol.PermissionToolInvoke},
				Class:               ToolRead, Idempotency: ToolInvocationIdempotent, TimeoutMilliseconds: 1000,
			},
			Handler: ToolHandlerFunc(func(context.Context, ToolInvocationContext, json.RawMessage) (ToolResult, error) {
				return ToolResult{Value: json.RawMessage(`{}`)}, nil
			}),
		})
	}
	return NewToolRegistry(tools, allowToolAuthorizer{})
}
