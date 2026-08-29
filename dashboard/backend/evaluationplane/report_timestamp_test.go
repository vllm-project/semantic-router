package evaluationplane

import (
	"errors"
	"testing"
	"time"
)

func TestReportExecutionTimestampSeparatesReplayFromLiveEvidence(t *testing.T) {
	createdAt := time.Date(2026, time.August, 29, 1, 2, 3, 123456000, time.UTC)
	startedAt := createdAt.Add(time.Second)
	sealedAt := startedAt.Add(time.Second)
	run := Run{StartedAt: &startedAt}

	for _, test := range []struct {
		name      string
		mode      Mode
		generated time.Time
		wantErr   bool
	}{
		{name: "replay deterministic creation timestamp", mode: ModeReplay, generated: createdAt},
		{name: "replay completion timestamp", mode: ModeReplay, generated: startedAt},
		{name: "replay predates manifest", mode: ModeReplay, generated: createdAt.Add(-time.Nanosecond), wantErr: true},
		{name: "live predates start", mode: ModeLive, generated: createdAt, wantErr: true},
		{name: "live starts at server transition", mode: ModeLive, generated: startedAt},
		{name: "future timestamp", mode: ModeLive, generated: sealedAt.Add(time.Nanosecond), wantErr: true},
		{name: "unknown mode", mode: Mode("unknown"), generated: startedAt, wantErr: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := validateReportExecutionTimestamp(
				run,
				RunManifest{Mode: test.mode, CreatedAt: createdAt},
				test.generated,
				sealedAt,
			)
			if test.wantErr && !errors.Is(err, ErrInvalid) {
				t.Fatalf("error=%v, want ErrInvalid", err)
			}
			if !test.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	if err := validateReportExecutionTimestamp(Run{}, RunManifest{Mode: ModeReplay, CreatedAt: createdAt}, createdAt, sealedAt); !errors.Is(err, ErrInvalid) {
		t.Fatalf("missing server start error=%v, want ErrInvalid", err)
	}
}
