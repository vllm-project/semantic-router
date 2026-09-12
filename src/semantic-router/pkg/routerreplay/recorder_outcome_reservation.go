package routerreplay

import "maps"

// OutcomeReservation owns space for at most one scheduled and one terminal
// event. Ordinary best-effort events cannot consume this space. Capacity remains
// occupied until the writer finishes the terminal storage operation, including
// when scheduled has already been written. Storage errors retain the recorder's
// existing failure reporting; reservation guarantees admission, not durability.
type OutcomeReservation struct {
	queue     *outcomeQueue
	recorder  *Recorder
	id        string
	scheduled bool
	finished  bool
}

// TryReserveOutcome never waits for storage or capacity. All recipe recorders
// share the runtime's limit of DefaultOutcomeQueueCapacity outstanding attempts.
// A successful caller must eventually call Finish, even if its work is rejected.
func (r *Recorder) TryReserveOutcome(id string) *OutcomeReservation {
	q := r.outcomes
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed || q.attempts == cap(q.reserved)/2 {
		return nil
	}
	q.attempts++
	q.startLocked()
	return &OutcomeReservation{queue: q, recorder: r, id: id}
}

func (q *outcomeQueue) startLocked() {
	if !q.started {
		q.started = true
		go q.run()
	}
}

// Scheduled publishes the optional initial state at most once.
func (r *OutcomeReservation) Scheduled(outcome Outcome) {
	r.publish(outcome, false)
}

// Finish publishes the terminal state exactly once. It remains available to
// existing reservations after drain starts; shutdown waits for their writer.
func (r *OutcomeReservation) Finish(outcome Outcome) {
	r.publish(outcome, true)
}

func (r *OutcomeReservation) publish(outcome Outcome, terminal bool) {
	q := r.queue
	q.mu.Lock()
	defer q.mu.Unlock()
	if r.finished || (!terminal && r.scheduled) {
		return
	}
	outcome.Metadata = maps.Clone(outcome.Metadata)
	event := queuedOutcome{recorder: r.recorder, id: r.id, outcome: outcome}
	if terminal {
		r.finished = true
		event.terminal = r
	} else {
		r.scheduled = true
	}
	// Every outstanding reservation can publish at most two events into a
	// 2*capacity buffer, so publication cannot wait for the storage worker.
	q.reserved <- event
}

func (r *OutcomeReservation) release() {
	q := r.queue
	q.mu.Lock()
	defer q.mu.Unlock()
	q.attempts--
	if q.closed && q.attempts == 0 {
		close(q.reserved)
	}
}
