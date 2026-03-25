package extproc

import (
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func makeTool(name string) openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name: name,
		},
	}
}

func TestFilterToolsByDecisionPolicy_AllowList(t *testing.T) {
	tools := []openai.ChatCompletionToolParam{
		makeTool("read_file"),
		makeTool("write_file"),
		makeTool("search_web"),
		makeTool("exec_cmd"),
	}

	filtered := filterToolsByDecisionPolicy(tools, []string{"read_file", "search_web"}, nil)

	assert.Len(t, filtered, 2)
	names := make(map[string]bool)
	for _, tool := range filtered {
		names[tool.Function.Name] = true
	}
	assert.True(t, names["read_file"])
	assert.True(t, names["search_web"])
	assert.False(t, names["write_file"])
	assert.False(t, names["exec_cmd"])
}

func TestFilterToolsByDecisionPolicy_BlockList(t *testing.T) {
	tools := []openai.ChatCompletionToolParam{
		makeTool("read_file"),
		makeTool("write_file"),
		makeTool("search_web"),
		makeTool("exec_cmd"),
	}

	filtered := filterToolsByDecisionPolicy(tools, nil, []string{"exec_cmd", "write_file"})

	assert.Len(t, filtered, 2)
	names := make(map[string]bool)
	for _, tool := range filtered {
		names[tool.Function.Name] = true
	}
	assert.True(t, names["read_file"])
	assert.True(t, names["search_web"])
	assert.False(t, names["write_file"])
	assert.False(t, names["exec_cmd"])
}

func TestFilterToolsByDecisionPolicy_AllowAndBlock(t *testing.T) {
	tools := []openai.ChatCompletionToolParam{
		makeTool("read_file"),
		makeTool("write_file"),
		makeTool("search_web"),
	}

	// Allow read_file and write_file, but block write_file
	filtered := filterToolsByDecisionPolicy(tools, []string{"read_file", "write_file"}, []string{"write_file"})

	assert.Len(t, filtered, 1)
	assert.Equal(t, "read_file", filtered[0].Function.Name)
}

func TestFilterToolsByDecisionPolicy_EmptyBoth(t *testing.T) {
	tools := []openai.ChatCompletionToolParam{
		makeTool("a"),
		makeTool("b"),
	}

	filtered := filterToolsByDecisionPolicy(tools, nil, nil)
	assert.Len(t, filtered, 2)
}

func TestFilterToolsByDecisionPolicy_EmptyTools(t *testing.T) {
	filtered := filterToolsByDecisionPolicy(nil, []string{"a"}, nil)
	assert.Empty(t, filtered)
}

func TestToolScopeConstants(t *testing.T) {
	assert.Equal(t, "none", config.ToolScopeNone)
	assert.Equal(t, "local_only", config.ToolScopeLocalOnly)
	assert.Equal(t, "standard", config.ToolScopeStandard)
	assert.Equal(t, "full", config.ToolScopeFull)
}

func TestDecisionToolScopeFields(t *testing.T) {
	dec := config.Decision{
		Name:       "test",
		ToolScope:  "local_only",
		AllowTools: []string{"read_file", "list_dir"},
		BlockTools: []string{"exec_cmd"},
	}

	assert.Equal(t, "test", dec.Name)
	assert.Equal(t, "local_only", dec.ToolScope)
	assert.Equal(t, []string{"read_file", "list_dir"}, dec.AllowTools)
	assert.Equal(t, []string{"exec_cmd"}, dec.BlockTools)
}
