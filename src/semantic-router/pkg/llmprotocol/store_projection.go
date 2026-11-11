package llmprotocol

// ProjectUnstoredRequest drops store: true from a request for a backend whose
// wire format keeps no completions. The caller applies it only when the
// operator lists LossyFieldStore and the target has no request storage, and
// must report the returned diagnostic.
func ProjectUnstoredRequest(request Request, target WireFormat) (Request, Diagnostics) {
	if request.Store == nil || !*request.Store {
		return request, nil
	}
	projected := request
	projected.Store = nil
	projected.Generation++ // The source envelope still carries store.
	return projected, Diagnostics{{
		Source: request.Trusted.SourceFormat,
		Target: target,
		Field:  LossyFieldStore,
		Action: DiagnosticDropped,
		Reason: "the selected backend keeps no stored completions",
	}}
}
