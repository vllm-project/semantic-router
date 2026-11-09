package store

// HistoryResetRecoveryStatus is the replay wire form of a history-reset
// recovery outcome. Replay owns this vocabulary so the store does not depend
// on the history-reset action and its context-transformation graph.
type HistoryResetRecoveryStatus string

const (
	HistoryResetRecoveryStored      HistoryResetRecoveryStatus = "stored"
	HistoryResetRecoveryNotRequired HistoryResetRecoveryStatus = "not_required"
	HistoryResetRecoveryFailed      HistoryResetRecoveryStatus = "failed"
)

// JSONEnumValues lists every value a recorded status can take on the wire,
// for generated API schemas. The empty status is not one of them: it is
// omitted rather than encoded.
func (HistoryResetRecoveryStatus) JSONEnumValues() []string {
	return []string{
		string(HistoryResetRecoveryNotRequired),
		string(HistoryResetRecoveryStored),
		string(HistoryResetRecoveryFailed),
	}
}
