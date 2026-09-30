package catalog

import "testing"

func TestReasoningTransportSupportsFamilyType(t *testing.T) {
	tests := []struct {
		transport  ReasoningTransport
		familyType string
		want       bool
	}{
		{ReasoningTransportChatTemplate, "reasoning_effort", true},
		{ReasoningTransportTopLevelEffort, "reasoning_effort", true},
		{ReasoningTransportTopLevelEffort, "reasoning_mode", false},
		{ReasoningTransportTopLevelBoolean, "chat_template_kwargs", true},
		{ReasoningTransportEffortTemplateSwitch, "reasoning_effort", true},
		{ReasoningTransportEffortTemplateSwitch, "chat_template_kwargs", false},
		{ReasoningTransportEffortBooleanSwitch, "reasoning_effort", true},
		{ReasoningTransportReasoningObject, "reasoning_mode", true},
		{ReasoningTransportThinkingObject, "reasoning_mode", true},
		{ReasoningTransportThinkingObject, "reasoning_effort", false},
		{ReasoningTransportThinkingEffort, "reasoning_effort", true},
		{ReasoningTransportOutputConfig, "reasoning_effort", true},
		{ReasoningTransportDeepSeekThinking, "top_level_reasoning_effort", true},
		{ReasoningTransport("invented"), "reasoning_effort", false},
	}
	for _, test := range tests {
		t.Run(string(test.transport)+"/"+test.familyType, func(t *testing.T) {
			if got := test.transport.SupportsFamilyType(test.familyType); got != test.want {
				t.Fatalf("SupportsFamilyType() = %v, want %v", got, test.want)
			}
		})
	}
}
