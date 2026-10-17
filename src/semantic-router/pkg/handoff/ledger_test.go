/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package handoff

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func ledgerEnvelope(t *testing.T, id string, overrides map[string]any) *Envelope {
	t.Helper()
	fields := map[string]any{"handoff_id": id}
	for key, value := range overrides {
		fields[key] = value
	}
	envelope, err := Parse(validJSON(fields), testNow)
	require.NoError(t, err)
	return envelope
}

func TestLedgerRetriesAreIdempotentAndReuseIsAConflict(t *testing.T) {
	ledger := NewLedger(8)
	original := ledgerEnvelope(t, "h-1", nil)

	assert.Equal(t, AdmissionNew, ledger.Admit(original, testNow))
	assert.Equal(t, AdmissionDuplicate, ledger.Admit(original, testNow.Add(time.Second)))
	assert.Equal(t, AdmissionDuplicate, ledger.Admit(ledgerEnvelope(t, "h-1", nil), testNow.Add(2*time.Second)))
	assert.Equal(t, AdmissionConflict, ledger.Admit(
		ledgerEnvelope(t, "h-1", map[string]any{"root_invocation_id": "inv-other"}),
		testNow.Add(3*time.Second),
	))
	assert.Equal(t, AdmissionDuplicate, ledger.Admit(original, testNow.Add(4*time.Second)), "a conflict must not replace the original record")
	assert.Equal(t, AdmissionNew, ledger.Admit(ledgerEnvelope(t, "h-2", nil), testNow))
}

func TestLedgerCancellationIsStickyUntilExpiry(t *testing.T) {
	ledger := NewLedger(8)
	active := ledgerEnvelope(t, "h-1", nil)
	cancel := ledgerEnvelope(t, "h-1", map[string]any{
		"state":      "cancelled",
		"expires_at": testNow.Add(10 * time.Minute).Format(time.RFC3339),
	})

	require.Equal(t, AdmissionNew, ledger.Admit(active, testNow))
	assert.Equal(t, AdmissionCancelRecorded, ledger.Admit(cancel, testNow.Add(time.Second)))
	assert.Equal(t, AdmissionCancelRecorded, ledger.Admit(cancel, testNow.Add(2*time.Second)), "repeated cancellation is idempotent")
	assert.Equal(t, AdmissionCancelled, ledger.Admit(active, testNow.Add(3*time.Second)))

	// The record lives until the later of the two expiries.
	assert.Equal(t, AdmissionCancelled, ledger.Admit(active, testNow.Add(9*time.Minute)))
	assert.Equal(t, AdmissionNew, ledger.Admit(
		ledgerEnvelope(t, "h-1", map[string]any{"expires_at": testNow.Add(15 * time.Minute).Format(time.RFC3339)}),
		testNow.Add(10*time.Minute),
	))
}

func TestLedgerCancellationBeforeFirstUseBlocksLaterRetries(t *testing.T) {
	ledger := NewLedger(8)

	assert.Equal(t, AdmissionCancelRecorded, ledger.Admit(ledgerEnvelope(t, "h-1", map[string]any{"state": "cancelled"}), testNow))
	assert.Equal(t, AdmissionCancelled, ledger.Admit(ledgerEnvelope(t, "h-1", nil), testNow))
}

func TestLedgerForgetsExpiredRecords(t *testing.T) {
	ledger := NewLedger(8)
	envelope := ledgerEnvelope(t, "h-1", nil)

	require.Equal(t, AdmissionNew, ledger.Admit(envelope, testNow))
	assert.Equal(t, AdmissionNew, ledger.Admit(envelope, envelope.ExpiresAt), "expiry is exclusive")
}

func TestLedgerIsBoundedAndEvictsOldestFirst(t *testing.T) {
	ledger := NewLedger(3)
	for i := range 5 {
		ledger.Admit(ledgerEnvelope(t, fmt.Sprintf("h-%d", i), nil), testNow)
	}

	assert.Equal(t, 3, ledger.Len())
	assert.Equal(t, AdmissionNew, ledger.Admit(ledgerEnvelope(t, "h-0", nil), testNow), "the oldest record was evicted")
	assert.Equal(t, AdmissionDuplicate, ledger.Admit(ledgerEnvelope(t, "h-4", nil), testNow))
}

func TestLedgerAdmitsOneFirstSightingUnderConcurrency(t *testing.T) {
	ledger := NewLedger(8)
	envelope := ledgerEnvelope(t, "h-1", nil)
	outcomes := make(chan Admission, 32)
	var wg sync.WaitGroup
	for range cap(outcomes) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			outcomes <- ledger.Admit(envelope, testNow)
		}()
	}
	wg.Wait()
	close(outcomes)

	counts := map[Admission]int{}
	for outcome := range outcomes {
		counts[outcome]++
	}
	assert.Equal(t, map[Admission]int{AdmissionNew: 1, AdmissionDuplicate: cap(outcomes) - 1}, counts)
}
