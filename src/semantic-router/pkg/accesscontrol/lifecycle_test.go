package accesscontrol

import "testing"

func TestTeamStatusContract(t *testing.T) {
	t.Parallel()

	for _, status := range []TeamStatus{TeamStatusActive, TeamStatusDisabled} {
		if !status.Valid() {
			t.Fatalf("expected Team status %q to be valid", status)
		}
	}
	for _, status := range []TeamStatus{"", "draft", "deleted"} {
		if status.Valid() {
			t.Fatalf("expected Team status %q to be rejected", status)
		}
	}
}
