package accesspublisher

import "testing"

func TestStateMachineEnforcesRestrictivePublicationOrder(t *testing.T) {
	state := State{Restrictive: true}
	want := []Action{
		ActionInstallBarriers, ActionStage, ActionValidate, ActionWaitBarrierAck,
		ActionWaitRoutingAck, ActionActivate, ActionCompact, ActionPersist,
		ActionMarkApplied, ActionClearBarriers,
	}
	for _, action := range want {
		if got := state.Next(); got != action {
			t.Fatalf("Next() = %q, want %q", got, action)
		}
		if err := state.Complete(action); err != nil {
			t.Fatalf("Complete(%q) error = %v", action, err)
		}
	}
	if state.Next() != ActionDone {
		t.Fatalf("final Next() = %q", state.Next())
	}
}

func TestStateMachineRejectsSkippedTransition(t *testing.T) {
	state := State{Restrictive: true}
	if err := state.Complete(ActionStage); err == nil {
		t.Fatal("state machine accepted stage before barriers")
	}
}

func TestExpansionSkipsOnlyRestrictionTransitions(t *testing.T) {
	state := State{}
	if state.Next() != ActionStage {
		t.Fatalf("expansion Next() = %q", state.Next())
	}
	state.Staged, state.Validated = true, true
	if state.Next() != ActionWaitRoutingAck {
		t.Fatalf("expansion acknowledgement Next() = %q", state.Next())
	}
}
