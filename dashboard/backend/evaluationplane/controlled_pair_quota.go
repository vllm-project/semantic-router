package evaluationplane

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// controlledPairIntentReservationBytes is the persistent aggregate envelope
// charged from admission through terminal publication. Every variable-size
// runtime field is represented at its validated maximum.
func controlledPairIntentReservationBytes(pair controlledPairManifest) (int64, error) {
	states := make([]controlledPairManifest, 0, 7)
	base := pair
	base.State = controlledPairStatePending
	base.BaselineStageName, base.CandidateStageName = "", ""
	base.StartedAt, base.DeletedAt = nil, nil
	base.StartReceiptDigest, base.DeleteReceiptDigest = "", ""
	for _, run := range []*Run{&base.BaselineRun, &base.CandidateRun} {
		run.Status, run.StartedAt, run.CompletedAt, run.Error = StatusPending, nil, nil, ""
		run.Progress.Percent, run.Progress.Completed, run.Progress.CurrentTrackID = 0, 0, ""
		run.Progress.Message = strings.Repeat("\x00", maxWorkerMessageBytes)
	}

	publishing := base
	publishing.State = controlledPairStatePublishing
	publishing.BaselineStageName = stagedRunBundlePrefix + strings.Repeat("0", 32)
	publishing.CandidateStageName = stagedRunBundlePrefix + strings.Repeat("1", 32)
	states = append(states, publishing)

	states = append(states, base)

	now := time.Unix(1_999_999_999, 999_999_999).UTC()
	starting := base
	starting.State, starting.StartedAt = controlledPairStateStarting, &now
	starting.StartReceiptDigest = digestString("controlled-pair-reserved-start-receipt")
	states = append(states, starting)

	running := starting
	running.State = controlledPairStateRunning
	running.BaselineRun = controlledPairRunningSnapshot(running.BaselineRun, now)
	running.CandidateRun = controlledPairRunningSnapshot(running.CandidateRun, now)
	states = append(states, running)

	terminal := base
	terminal.State, terminal.StartedAt = controlledPairStateTerminal, &now
	terminal.StartReceiptDigest = digestString("controlled-pair-reserved-start-receipt")
	for _, run := range []*Run{&terminal.BaselineRun, &terminal.CandidateRun} {
		run.Status, run.StartedAt, run.CompletedAt = StatusFailed, &now, &now
		run.Progress.Percent = 99.99999999999999
		run.Progress.Completed = run.Progress.Total
		for _, trackID := range run.TrackIDs {
			if len(trackID) > len(run.Progress.CurrentTrackID) {
				run.Progress.CurrentTrackID = trackID
			}
		}
		// A single-byte control character has the maximum JSON expansion
		// ("\\u0000"). Validation limits raw bytes, so the envelope must reserve
		// the encoded representation rather than an ASCII approximation.
		run.Progress.Message = strings.Repeat("\x00", maxWorkerMessageBytes)
		run.Error = strings.Repeat("\x00", maxWorkerMessageBytes)
	}
	states = append(states, terminal)

	cancelling := running
	cancelling.State = controlledPairStateCancelling
	states = append(states, cancelling)

	deleting := terminal
	deleting.State, deleting.DeletedAt = controlledPairStateDeleting, &now
	deleting.DeleteReceiptDigest = digestString("controlled-pair-reserved-delete-receipt")
	states = append(states, deleting)

	var maximum int
	for _, state := range states {
		encoded, err := json.MarshalIndent(state, "", "  ")
		if err != nil {
			return 0, fmt.Errorf("encode controlled pair quota reservation: %w", err)
		}
		if len(encoded)+1 > maximum {
			maximum = len(encoded) + 1
		}
	}
	return int64(maximum), nil
}

func validateControlledPairAggregateReservation(pair controlledPairManifest) error {
	reservation, err := controlledPairIntentReservationBytes(pair)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(pair, "", "  ")
	if err != nil {
		return fmt.Errorf("encode controlled pair aggregate: %w", err)
	}
	if int64(len(encoded)+1) > reservation {
		return fmt.Errorf("%w: controlled pair aggregate exceeds its durable reservation", ErrQuota)
	}
	return nil
}
