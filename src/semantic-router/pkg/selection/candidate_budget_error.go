package selection

// RequestBudgetError identifies a deterministic request-budget rejection. It
// remains a selection failure, but is distinct from missing capabilities,
// unknown limits, unavailable quality evidence or an empty model inventory.
type RequestBudgetError struct {
	Code    string
	Message string
}

func (err *RequestBudgetError) Error() string { return err.Message }

func (err *RequestBudgetError) Unwrap() error { return ErrNoEligibleCandidates }
