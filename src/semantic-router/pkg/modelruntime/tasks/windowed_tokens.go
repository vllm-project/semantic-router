package tasks

// WindowedTokenClassification carries one global span result plus exact native
// coverage. Overlap work is not double-counted as processed document tokens.
type WindowedTokenClassification struct {
	Result        TokenClassificationResult
	ContentTokens int
	Windows       [][2]int
}

func (r WindowedTokenClassification) InputMetadata() *InputUsage { return r.Result.Input }
