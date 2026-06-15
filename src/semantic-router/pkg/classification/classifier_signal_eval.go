package classification

// EvaluateAllSignals evaluates all signal types and returns SignalResults
// This is the new method that includes fact_check signals
func (c *Classifier) EvaluateAllSignals(text string) *SignalResults {
	return c.EvaluateAllSignalsWithContext(text, text, text, nil, nil, false, false, "", nil, ConversationFacts{})
}

// EvaluateAllSignalsWithForceOption evaluates signals with option to force evaluate all
// forceEvaluateAll: if true, evaluates all configured signals regardless of decision usage
func (c *Classifier) EvaluateAllSignalsWithForceOption(text string, forceEvaluateAll bool) *SignalResults {
	return c.EvaluateAllSignalsWithContext(text, text, text, nil, nil, false, forceEvaluateAll, "", nil, ConversationFacts{})
}
