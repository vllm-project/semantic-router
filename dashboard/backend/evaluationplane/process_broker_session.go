package evaluationplane

import (
	"context"
	"errors"
	"sync"
)

var (
	errWorkerBrokerResponseBudget = errors.New("evaluation worker HTTP broker response budget exceeded")
	errWorkerBrokerLedgerReplay   = errors.New("evaluation worker HTTP broker method ledger was requested more than once")
)

// workerBrokerSessionState owns the two session-wide admission boundaries
// that are shared by every concurrent broker request. Its mutex is never held
// while acquiring the broker write, model, or transcript locks. Cancellation
// is copied under the mutex and invoked only after unlocking.
type workerBrokerSessionState struct {
	mu sync.Mutex

	responseBytes int64
	responseLimit int64
	ledgerReads   map[string]struct{}
	cancel        context.CancelFunc
	terminalErr   error
}

func newWorkerBrokerSessionState(responseLimit int64) workerBrokerSessionState {
	return workerBrokerSessionState{
		responseLimit: responseLimit,
		ledgerReads:   make(map[string]struct{}, 4),
	}
}

func (broker *workerHTTPBroker) bindSessionCancellation(cancel context.CancelFunc) {
	broker.session.mu.Lock()
	broker.session.cancel = cancel
	terminal := broker.session.terminalErr != nil
	broker.session.mu.Unlock()
	if terminal {
		cancel()
	}
}

func (broker *workerHTTPBroker) releaseSessionCancellation() {
	broker.session.mu.Lock()
	broker.session.cancel = nil
	broker.session.mu.Unlock()
}

func (broker *workerHTTPBroker) reserveResponseBytes(size int64) error {
	broker.session.mu.Lock()
	if broker.session.terminalErr != nil {
		err := broker.session.terminalErr
		broker.session.mu.Unlock()
		return err
	}
	if size < 0 || broker.session.responseBytes > broker.session.responseLimit ||
		size > broker.session.responseLimit-broker.session.responseBytes {
		broker.session.terminalErr = errWorkerBrokerResponseBudget
		cancel := broker.session.cancel
		broker.session.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return errWorkerBrokerResponseBudget
	}
	broker.session.responseBytes += size
	broker.session.mu.Unlock()
	return nil
}

func (broker *workerHTTPBroker) admitRequest(request workerBrokerRequest) error {
	broker.session.mu.Lock()
	if broker.session.terminalErr != nil {
		err := broker.session.terminalErr
		broker.session.mu.Unlock()
		return err
	}
	if !isMethodLedgerOperation(request.Operation) {
		broker.session.mu.Unlock()
		return nil
	}
	if _, duplicate := broker.session.ledgerReads[request.Operation]; duplicate {
		broker.session.terminalErr = errWorkerBrokerLedgerReplay
		cancel := broker.session.cancel
		broker.session.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return errWorkerBrokerLedgerReplay
	}
	broker.session.ledgerReads[request.Operation] = struct{}{}
	broker.session.mu.Unlock()
	return nil
}

func (broker *workerHTTPBroker) abortSession(err error) {
	if err == nil {
		return
	}
	broker.session.mu.Lock()
	if broker.session.terminalErr == nil {
		broker.session.terminalErr = err
	}
	cancel := broker.session.cancel
	broker.session.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (broker *workerHTTPBroker) sessionFailure() error {
	broker.session.mu.Lock()
	defer broker.session.mu.Unlock()
	return broker.session.terminalErr
}

func (broker *workerHTTPBroker) sessionResponseUsage() (used, limit int64) {
	broker.session.mu.Lock()
	defer broker.session.mu.Unlock()
	return broker.session.responseBytes, broker.session.responseLimit
}

func methodLedgerRequestIdentity(operation string) (TrackID, string, bool) {
	switch operation {
	case workerBrokerAgentTaskLedger:
		return "agentic", "agent-task-ledger", true
	case workerBrokerFaultRecoveryLedger:
		return "agentic", "fault-recovery-ledger", true
	case workerBrokerHardPolicyLedger:
		return "safety", "hard-policy-ledger", true
	case workerBrokerProductionExperimentLedger:
		return "preference", "production-ledger", true
	default:
		return "", "", false
	}
}

func validateMethodLedgerRequestIdentity(request workerBrokerRequest) error {
	trackID, caseID, ok := methodLedgerRequestIdentity(request.Operation)
	if !ok || request.TrackID != trackID || request.CaseID != caseID || request.AttemptID != "ledger-fetch" {
		return errors.New("evaluation worker HTTP broker method ledger identity is invalid")
	}
	return nil
}
