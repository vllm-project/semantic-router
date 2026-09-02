package evaluationplane

import "fmt"

// validateAgenticSuiteEndpoints keeps task-quality and G6 recovery evidence
// on distinct provider ledgers. Neither endpoint can qualify the other method.
func validateAgenticSuiteEndpoints(
	mode Mode,
	suiteIDs []string,
	agentTaskLedger *ServiceEndpoint,
	faultRecoveryLedger *ServiceEndpoint,
) error {
	if mode != ModeLive {
		return nil
	}
	for _, suiteID := range suiteIDs {
		switch suiteID {
		case "live-agent-tasks":
			if agentTaskLedger == nil {
				return fmt.Errorf("%w: live-agent-tasks requires its dedicated agent_task_ledger capability", ErrInvalid)
			}
		case "live-fault-recovery":
			if faultRecoveryLedger == nil {
				return fmt.Errorf("%w: live-fault-recovery requires its dedicated fault_recovery_ledger capability", ErrInvalid)
			}
		}
	}
	return nil
}
