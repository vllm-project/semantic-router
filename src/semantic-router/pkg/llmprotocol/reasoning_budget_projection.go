package llmprotocol

// ProjectUnsupportedReasoningBudget removes a neutral Chat budget for a
// provider that cannot honor it. The caller applies this only when the
// operator allows LossyFieldReasoningBudget and reports the diagnostic.
func ProjectUnsupportedReasoningBudget(request Request, target WireFormat) (Request, Diagnostics) {
	if request.ReasoningBudgetTokens == nil {
		return request, nil
	}
	projected := request
	projected.ReasoningBudgetTokens = nil
	if projected.ReasoningMode == ReasoningModeEnabled {
		// Chat has no portable enabled switch without the token budget. Let the
		// provider's default reasoning behavior apply once that budget is lost.
		projected.ReasoningMode = ""
	}
	projected.Generation++ // The source envelope still carries the budget.
	return projected, Diagnostics{{
		Source: request.Trusted.SourceFormat,
		Target: target,
		Field:  LossyFieldReasoningBudget,
		Action: DiagnosticDropped,
		Reason: "the selected Chat provider cannot honor the budget, so its default reasoning behavior applies",
	}}
}
