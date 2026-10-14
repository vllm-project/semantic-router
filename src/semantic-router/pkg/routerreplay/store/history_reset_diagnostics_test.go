package store

import (
	"encoding/json"
	"strings"
	"testing"
)

// Recovery status is a closed set on the wire, and an unset status or zero
// entries are omitted rather than encoded as empty values.
func TestHistoryResetRecoveryFieldsEncodeTheClosedSet(t *testing.T) {
	cases := map[HistoryResetRecoveryStatus]string{
		HistoryResetRecoveryNotRequired: `"recovery_status":"not_required"`,
		HistoryResetRecoveryStored:      `"recovery_status":"stored"`,
		HistoryResetRecoveryFailed:      `"recovery_status":"failed"`,
	}
	for status, want := range cases {
		encoded, err := json.Marshal(HistoryResetDiagnostics{RecoveryStatus: status, RecoveryEntries: 1})
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(encoded), want) || !strings.Contains(string(encoded), `"recovery_entries":1`) {
			t.Fatalf("%s encoded as %s", status, encoded)
		}
	}
	if len(cases) != len(HistoryResetRecoveryStatus("").JSONEnumValues()) {
		t.Fatal("the test no longer covers every declared status")
	}

	encoded, err := json.Marshal(HistoryResetDiagnostics{Outcome: "skipped"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "recovery_") {
		t.Fatalf("an unset recovery status must be omitted: %s", encoded)
	}
}

// A cloned record owns its history-reset receipt, like every other receipt
// the clone copies: changing the clone must not change the stored original.
func TestCloneRouteDiagnosticsCopiesTheHistoryResetReceipt(t *testing.T) {
	original := &RouteDiagnostics{HistoryReset: &HistoryResetDiagnostics{
		Outcome: "applied", RemovedMessages: 2, RecoveryStatus: HistoryResetRecoveryStored,
	}}
	cloned := cloneRouteDiagnostics(original)
	if cloned.HistoryReset == nil || *cloned.HistoryReset != *original.HistoryReset {
		t.Fatalf("clone lost the receipt: %+v", cloned.HistoryReset)
	}
	cloned.HistoryReset.RemovedMessages = 0
	if original.HistoryReset.RemovedMessages != 2 {
		t.Fatal("the clone shares the original's receipt")
	}
}
