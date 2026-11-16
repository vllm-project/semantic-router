package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func makeTool(name string) llmprotocol.Tool {
	return llmprotocol.Tool{Name: name, InputSchema: []byte(`{"type":"object"}`)}
}

func TestFilterToolsByPolicy_AllowList(t *testing.T) {
	tools := []llmprotocol.Tool{
		makeTool("read_file"),
		makeTool("write_file"),
		makeTool("search_web"),
		makeTool("exec_cmd"),
	}

	filtered := filterToolsByPolicy(tools, filteredPolicy([]string{"read_file", "search_web"}, nil))

	assert.Len(t, filtered, 2)
	names := make(map[string]bool)
	for _, tool := range filtered {
		names[tool.Name] = true
	}
	assert.True(t, names["read_file"])
	assert.True(t, names["search_web"])
	assert.False(t, names["write_file"])
	assert.False(t, names["exec_cmd"])
}

func TestFilterToolsByPolicy_BlockList(t *testing.T) {
	tools := []llmprotocol.Tool{
		makeTool("read_file"),
		makeTool("write_file"),
		makeTool("search_web"),
		makeTool("exec_cmd"),
	}

	filtered := filterToolsByPolicy(tools, filteredPolicy(nil, []string{"exec_cmd", "write_file"}))

	assert.Len(t, filtered, 2)
	names := make(map[string]bool)
	for _, tool := range filtered {
		names[tool.Name] = true
	}
	assert.True(t, names["read_file"])
	assert.True(t, names["search_web"])
	assert.False(t, names["write_file"])
	assert.False(t, names["exec_cmd"])
}

func TestFilterToolsByPolicy_AllowAndBlock(t *testing.T) {
	tools := []llmprotocol.Tool{
		makeTool("read_file"),
		makeTool("write_file"),
		makeTool("search_web"),
	}

	// Allow read_file and write_file, but block write_file
	filtered := filterToolsByPolicy(tools, filteredPolicy([]string{"read_file", "write_file"}, []string{"write_file"}))

	assert.Len(t, filtered, 1)
	assert.Equal(t, "read_file", filtered[0].Name)
}

func TestFilterToolsByPolicy_EmptyBoth(t *testing.T) {
	tools := []llmprotocol.Tool{
		makeTool("a"),
		makeTool("b"),
	}

	filtered := filterToolsByPolicy(tools, filteredPolicy(nil, nil))
	assert.Len(t, filtered, 2)
}

func TestFilterToolsByPolicy_EmptyTools(t *testing.T) {
	filtered := filterToolsByPolicy(nil, filteredPolicy([]string{"a"}, nil))
	assert.Empty(t, filtered)
}

func filteredPolicy(allow, block []string) *config.ToolsPolicyConfig {
	return &config.ToolsPolicyConfig{Mode: config.ToolsPolicyModeFiltered, AllowTools: allow, BlockTools: block}
}

func TestDecisionGetToolsConfig(t *testing.T) {
	payload, err := config.NewStructuredPayload(config.ToolsPluginConfig{
		Enabled: true,
		Policy:  filteredPolicy([]string{"read_file", "list_dir"}, []string{"exec_cmd"}),
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
		assert.Equal(t, config.ToolsPolicyModeFiltered, cfg.Policy.EffectiveMode())
		assert.Equal(t, []string{"read_file", "list_dir"}, cfg.Policy.AllowTools)
		assert.Equal(t, []string{"exec_cmd"}, cfg.Policy.BlockTools)
	}
}

func TestClearToolChoiceWhenNoTools_RemovesAutoChoice(t *testing.T) {
	req := testNeutralRequest("test-model", "你好")
	req.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}

	changed := clearSemanticToolChoiceWhenNoTools(req)

	assert.True(t, changed)
	assert.Equal(t, llmprotocol.ToolChoice{}, req.ToolChoice)
}

func TestClearToolChoiceWhenNoTools_KeepsChoiceWhenToolsPresent(t *testing.T) {
	req := testNeutralRequest("test-model", "天气如何")
	req.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}
	req.Tools = []llmprotocol.Tool{makeTool("lookup_weather")}

	changed := clearSemanticToolChoiceWhenNoTools(req)

	assert.False(t, changed)
	assert.Equal(t, llmprotocol.ToolChoiceAuto, req.ToolChoice.Mode)
}

// A hosted image_generation operation does not occupy request.Tools, but it is
// still a tool: an explicit tool_choice: none forbidding all tools must be
// preserved, not erased, so capability dispatch can avoid the images backend.
func TestClearToolChoiceWhenNoTools_KeepsExplicitNoneWithHostedImageGeneration(t *testing.T) {
	req := testNeutralRequest("test-model", "画一只猫")
	req.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
	req.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNone}

	changed := clearSemanticToolChoiceWhenNoTools(req)

	assert.False(t, changed)
	assert.Equal(t, llmprotocol.ToolChoiceNone, req.ToolChoice.Mode)
}
