package accesspublisher

import "fmt"

type Action string

const (
	ActionInstallBarriers Action = "install_barriers"
	ActionStage           Action = "stage"
	ActionValidate        Action = "validate"
	ActionWaitBarrierAck  Action = "wait_barrier_ack"
	ActionWaitRoutingAck  Action = "wait_routing_ack"
	ActionActivate        Action = "activate"
	ActionCompact         Action = "compact"
	ActionPersist         Action = "persist"
	ActionMarkApplied     Action = "mark_applied"
	ActionClearBarriers   Action = "clear_barriers"
	ActionDone            Action = "done"
)

// State is the pure publication state machine. External adapters may retry any
// action, but cannot skip the barrier, validation, acknowledgement, or durable
// watermark ordering encoded here.
type State struct {
	Restrictive       bool
	BarriersInstalled bool
	Staged            bool
	Validated         bool
	BarrierAcked      bool
	RoutingAcked      bool
	Activated         bool
	Compacted         bool
	Persisted         bool
	AppliedMarked     bool
	BarriersCleared   bool
}

func (s State) Next() Action {
	if s.Restrictive && !s.BarriersInstalled {
		return ActionInstallBarriers
	}
	if !s.Staged {
		return ActionStage
	}
	if !s.Validated {
		return ActionValidate
	}
	if s.Restrictive && !s.BarrierAcked {
		return ActionWaitBarrierAck
	}
	if !s.RoutingAcked {
		return ActionWaitRoutingAck
	}
	if !s.Activated {
		return ActionActivate
	}
	if !s.Compacted {
		return ActionCompact
	}
	if !s.Persisted {
		return ActionPersist
	}
	if !s.AppliedMarked {
		return ActionMarkApplied
	}
	if s.Restrictive && !s.BarriersCleared {
		return ActionClearBarriers
	}
	return ActionDone
}

func (s *State) Complete(action Action) error {
	if s.Next() != action {
		return fmt.Errorf("publication action %q is out of order; next action is %q", action, s.Next())
	}
	switch action {
	case ActionInstallBarriers:
		s.BarriersInstalled = true
	case ActionStage:
		s.Staged = true
	case ActionValidate:
		s.Validated = true
	case ActionWaitBarrierAck:
		s.BarrierAcked = true
	case ActionWaitRoutingAck:
		s.RoutingAcked = true
	case ActionActivate:
		s.Activated = true
	case ActionCompact:
		s.Compacted = true
	case ActionPersist:
		s.Persisted = true
	case ActionMarkApplied:
		s.AppliedMarked = true
	case ActionClearBarriers:
		s.BarriersCleared = true
	case ActionDone:
		return fmt.Errorf("completed publication has no next transition")
	default:
		return fmt.Errorf("unknown publication action %q", action)
	}
	return nil
}
