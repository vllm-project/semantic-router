package handlers

import "testing"

// Temporary: proves #2793's gate fails CI on a failing backend test. Reverted in
// the next commit; see the PR description for the red and green run links.
func TestGateProof(t *testing.T) {
	t.Fatal("gate proof for #2793: this test must fail the Dashboard gate")
}
