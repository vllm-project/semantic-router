package extproc

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// A decision that turns thinking off on a Messages backend sends no effort, so
// the client's per-message effort changes go as its top-level effort does. A
// system message keeps its text. The changes stay when thinking is on, and
// when the family has no off control and the backend runs its default mode.
func TestMessagesThinkingOffDropsPerMessageEfforts(t *testing.T) {
	const body = `{"model":"test-model","max_tokens":64,%s"messages":[` +
		`{"role":"user","content":"Plan the migration."},` +
		`{"role":"system","content":[],"output_config":{"effort":"low"}},` +
		`{"role":"assistant","content":"Export, create the schema, import."},` +
		`{"role":"user","content":"Write the export step."},` +
		`{"role":"system","content":"Answer in one paragraph.","output_config":{"effort":"xhigh"}}]}`
	sent := []string{"", "low", "", "", "xhigh"}
	for _, test := range []struct {
		name    string
		modes   []string
		client  string
		enabled bool
		want    []string
	}{
		{"thinking off", []string{config.ReasoningModeAdaptive, config.ReasoningModeDisabled}, "", false, []string{"", "", "", ""}},
		{
			"thinking off as the client sent it",
			[]string{config.ReasoningModeAdaptive, config.ReasoningModeDisabled},
			`"thinking":{"type":"disabled"},`, false,
			[]string{"", "", "", ""},
		},
		{"thinking on", []string{config.ReasoningModeAdaptive, config.ReasoningModeDisabled}, "", true, sent},
		{"no off control", []string{config.ReasoningModeAdaptive}, "", false, sent},
	} {
		t.Run(test.name, func(t *testing.T) {
			router := routingTestRouter("test-model")
			model := router.Config.ModelConfig["test-model"]
			model.ReasoningFamily = "claude"
			router.Config.ModelConfig["test-model"] = model
			router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
				"claude": {Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "effort", Modes: test.modes},
			}
			request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(
				llmprotocol.AnthropicMessagesV1, []byte(fmt.Sprintf(body, test.client)))
			require.NoError(t, err)
			clientMessages := request.Messages
			useReasoning := test.enabled
			decision := &config.Decision{Name: "agent", ModelRefs: []config.ModelRef{{
				Model: "test-model", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &useReasoning},
			}}}

			changed := router.applySemanticReasoningMode(
				&request, "test-model", llmprotocol.AnthropicMessagesV1, test.enabled, decision)

			assert.Equal(t, test.want, messageEfforts(request.Messages))
			last := request.Messages[len(request.Messages)-1]
			assert.Equal(t, llmprotocol.RoleSystem, last.Role)
			require.Len(t, last.Content, 1)
			assert.Equal(t, "Answer in one paragraph.", last.Content[0].Text)
			// Another candidate may still read the messages the client sent.
			assert.Equal(t, sent, messageEfforts(clientMessages))
			if len(test.want) < len(sent) {
				assert.True(t, changed, "the backend request must be encoded again")
			}
		})
	}
}

func messageEfforts(messages []llmprotocol.Message) []string {
	efforts := make([]string, 0, len(messages))
	for _, message := range messages {
		efforts = append(efforts, message.ReasoningEffort)
	}
	return efforts
}
