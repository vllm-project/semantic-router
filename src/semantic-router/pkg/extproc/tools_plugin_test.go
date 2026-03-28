package extproc

import (
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
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

func TestToolsPluginModeConstants(t *testing.T) {
	assert.Equal(t, "none", config.ToolsPluginModeNone)
	assert.Equal(t, "passthrough", config.ToolsPluginModePassthrough)
	assert.Equal(t, "filtered", config.ToolsPluginModeFiltered)
}

func TestDecisionGetToolsConfig(t *testing.T) {
	payload, err := config.NewStructuredPayload(config.ToolsPluginConfig{
		Enabled:    true,
		Mode:       config.ToolsPluginModeFiltered,
		AllowTools: []string{"read_file", "list_dir"},
		BlockTools: []string{"exec_cmd"},
	})
	assert.NoError(t, err)

	dec := config.Decision{
		Name: "test",
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginTools, Configuration: payload},
		},
	}

	cfg := dec.GetToolsConfig()
	if assert.NotNil(t, cfg) {
		assert.Equal(t, config.ToolsPluginModeFiltered, cfg.EffectiveMode())
		assert.Equal(t, []string{"read_file", "list_dir"}, cfg.AllowTools)
		assert.Equal(t, []string{"exec_cmd"}, cfg.BlockTools)
	}
}

func TestClearToolChoiceWhenNoTools_RemovesAutoChoice(t *testing.T) {
	requestJSON := []byte(`{
		"model": "test-model",
		"messages": [{"role": "user", "content": "你好"}],
		"tool_choice": "auto"
	}`)

	req, err := parseOpenAIRequest(requestJSON)
	assert.NoError(t, err)

	changed := clearToolChoiceWhenNoTools(req)

	assert.True(t, changed)
	assert.True(t, param.IsOmitted(req.ToolChoice.OfAuto))
	assert.Nil(t, req.ToolChoice.OfChatCompletionNamedToolChoice)
}

func TestClearToolChoiceWhenNoTools_KeepsChoiceWhenToolsPresent(t *testing.T) {
	requestJSON := []byte(`{
		"model": "test-model",
		"messages": [{"role": "user", "content": "天气如何"}],
		"tool_choice": "auto",
		"tools": [{
			"type": "function",
			"function": {"name": "lookup_weather"}
		}]
	}`)

	req, err := parseOpenAIRequest(requestJSON)
	assert.NoError(t, err)

	changed := clearToolChoiceWhenNoTools(req)

	assert.False(t, changed)
	assert.False(t, param.IsOmitted(req.ToolChoice.OfAuto))
}
