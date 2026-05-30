package extproc

func deriveAgenticContextPortability(
	reqCtx *RequestContext,
	hasMemory bool,
) (int, float64, bool, bool) {
	if reqCtx == nil {
		return 0, 1, false, false
	}
	portableHistoryTokens := reqCtx.HistoryTokenCount
	contextTokens := reqCtx.VSRContextTokenCount
	if contextTokens < portableHistoryTokens {
		contextTokens = portableHistoryTokens
	}
	continuationObserved := reqCtx.PreviousResponseID != "" ||
		reqCtx.PreviousModel != "" ||
		reqCtx.TurnIndex > 0 ||
		hasMemory ||
		portableHistoryTokens > 0
	if !continuationObserved {
		return portableHistoryTokens, 1, false, false
	}
	portability := 0.0
	if contextTokens > 0 {
		portability = float64(portableHistoryTokens) / float64(contextTokens)
		if portability > 1 {
			portability = 1
		}
	}
	providerStateOnly := reqCtx.PreviousResponseID != "" && portableHistoryTokens == 0
	return portableHistoryTokens, portability, true, providerStateOnly
}
