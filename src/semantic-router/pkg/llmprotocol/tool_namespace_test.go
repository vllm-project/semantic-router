package llmprotocol

import (
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

var portableToolNamePattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

func TestFlattenToolNameJoinsShortPortableNames(t *testing.T) {
	require.Equal(t, "multi_agent_v1__spawn_agent", FlattenToolName("multi_agent_v1", "spawn_agent"))
	require.Equal(t, "mcp__echo__echo", FlattenToolName("mcp__echo", "echo"))
}

func TestFlattenToolNameStaysPortableAndDistinct(t *testing.T) {
	long := strings.Repeat("server", 20)
	tests := []struct {
		name            string
		namespace, tool string
		other           [2]string
	}{
		{"long namespace", "mcp__" + long, "search", [2]string{"mcp__" + long + "x", "search"}},
		{"long tool", "mcp__docs", long, [2]string{"mcp__docs", long + "x"}},
		{"both long", long, long, [2]string{long + "x", long}},
		{"characters a backend rejects", "mcp__my.server", "get/issue", [2]string{"mcp__my_server", "get_issue"}},
		{"multibyte", "命名空间", "工具", [2]string{"命名空间", "工具二"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			flattened := FlattenToolName(test.namespace, test.tool)
			require.Regexp(t, portableToolNamePattern, flattened)
			require.Equal(t, flattened, FlattenToolName(test.namespace, test.tool), "a later turn must flatten to the same name")
			require.NotEqual(t, flattened, FlattenToolName(test.other[0], test.other[1]))
		})
	}
}

func TestFlattenToolNameKeepsTheToolNameReadableWhenShortened(t *testing.T) {
	flattened := FlattenToolName("mcp__"+strings.Repeat("server", 20), "search_issues")
	require.True(t, strings.HasSuffix(flattened, "__search_issues"), flattened)
}

func TestToolNamespacesRestoreOnlyDeclaredNames(t *testing.T) {
	namespaces := RequestToolNamespaces([]Tool{
		{Name: "exec_command"},
		{Name: "mcp__a__b__c", Namespace: "mcp__a__b", LocalName: "c"},
	})
	restored := namespaces.RestoreCall(&ToolCall{ID: "call_1", Name: "mcp__a__b__c", Arguments: "{}"})
	require.Equal(t, "mcp__a__b", restored.Namespace)
	require.Equal(t, "c", restored.Name)
	require.Equal(t, "{}", restored.Arguments)

	plain := &ToolCall{ID: "call_2", Name: "exec_command"}
	require.Same(t, plain, namespaces.RestoreCall(plain))
	require.Nil(t, namespaces.RestoreCall(nil))
	require.Empty(t, RequestToolNamespaces([]Tool{{Name: "exec_command"}}))
}

func TestToolNamespacesRestoreResponseWithoutMutatingTheDecodedCall(t *testing.T) {
	namespaces := RequestToolNamespaces([]Tool{{Name: "ns__tool", Namespace: "ns", LocalName: "tool"}})
	call := &ToolCall{ID: "call_1", Name: "ns__tool"}
	alternative := &ToolCall{ID: "call_2", Name: "ns__tool"}
	response := Response{
		Output:       []OutputItem{{Content: []Content{{Kind: ContentToolCall, ToolCall: call}, {Kind: ContentText, Text: "text"}}}},
		Alternatives: [][]OutputItem{{{Content: []Content{{Kind: ContentToolCall, ToolCall: alternative}}}}},
	}
	namespaces.RestoreResponse(&response)
	require.Equal(t, "ns", response.Output[0].Content[0].ToolCall.Namespace)
	require.Equal(t, "tool", response.Output[0].Content[0].ToolCall.Name)
	require.Equal(t, "ns", response.Alternatives[0][0].Content[0].ToolCall.Namespace)
	require.Equal(t, "ns__tool", call.Name)
}
