package llmprotocol

import "testing"

func TestProjectUnsupportedReasoningBudgetDropsOnlyTheBudget(t *testing.T) {
	budget := int64(1024)
	request := Request{
		ReasoningEffort:       "high",
		ReasoningBudgetTokens: &budget,
		ReasoningMode:         ReasoningModeEnabled,
		Trusted:               TrustedMetadata{SourceFormat: AnthropicMessagesV1},
	}
	projected, diagnostics := ProjectUnsupportedReasoningBudget(request, OpenAIChatV1)
	if projected.ReasoningBudgetTokens != nil || projected.ReasoningEffort != request.ReasoningEffort ||
		projected.ReasoningMode != "" || projected.Generation != request.Generation+1 ||
		request.ReasoningBudgetTokens == nil || *request.ReasoningBudgetTokens != budget {
		t.Fatalf("projected request = %+v, original = %+v", projected, request)
	}
	if len(diagnostics) != 1 || diagnostics[0] != (Diagnostic{
		Source: AnthropicMessagesV1, Target: OpenAIChatV1,
		Field: LossyFieldReasoningBudget, Action: DiagnosticDropped,
		Reason: "the selected Chat provider cannot honor the budget, so its default reasoning behavior applies",
	}) {
		t.Fatalf("diagnostics = %+v", diagnostics)
	}

	for _, request := range []Request{{}, {ReasoningEffort: "high"}} {
		unchanged, diagnostics := ProjectUnsupportedReasoningBudget(request, OpenAIChatV1)
		if unchanged.ReasoningBudgetTokens != nil || unchanged.Generation != request.Generation || len(diagnostics) != 0 {
			t.Fatalf("budget-free request was projected: %+v %+v", unchanged, diagnostics)
		}
	}
}
