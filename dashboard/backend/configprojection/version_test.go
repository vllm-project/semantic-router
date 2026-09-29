package configprojection

import (
	"strings"
	"sync"
	"testing"
	"time"
)

func TestNewActivationVersionUsesNanosecondSuffix(t *testing.T) {
	t.Parallel()

	version := NewActivationVersion()
	parts := strings.Split(version, ".")
	if len(parts) != 2 {
		t.Fatalf("expected version with nanosecond suffix, got %q", version)
	}
	if len(parts[0]) != len("20060102-150405") {
		t.Fatalf("unexpected timestamp prefix in %q", version)
	}
	if len(parts[1]) != 9 {
		t.Fatalf("expected 9-digit nanosecond suffix, got %q", parts[1])
	}
}

func TestNewActivationVersionDiffersWithinSameSecond(t *testing.T) {
	t.Parallel()

	first := NewActivationVersion()
	second := NewActivationVersion()
	if first == second {
		t.Fatalf("expected unique activation versions, both %q", first)
	}
}

func TestActivationVersionsAdvanceWhenWallClockRepeatsOrRecedes(t *testing.T) {
	t.Parallel()
	clock := &activationVersionClock{}
	now := time.Date(2026, 9, 13, 22, 27, 47, 552242000, time.UTC)
	previous := clock.next(now)
	for _, stamp := range []time.Time{now, now.Add(-time.Second), now} {
		version := clock.next(stamp)
		if version <= previous {
			t.Fatalf("activation version did not advance: %q after %q", version, previous)
		}
		previous = version
	}
}

func TestActivationVersionsReserveConcurrentCallsOnSameClockTick(t *testing.T) {
	t.Parallel()
	clock := &activationVersionClock{}
	now := time.Date(2026, 9, 13, 22, 27, 47, 552242000, time.UTC)
	const calls = 256
	versions := make(chan string, calls)
	var workers sync.WaitGroup
	for range calls {
		workers.Add(1)
		go func() {
			defer workers.Done()
			versions <- clock.next(now)
		}()
	}
	workers.Wait()
	close(versions)
	seen := make(map[string]bool, calls)
	for version := range versions {
		if seen[version] {
			t.Fatalf("concurrent activation version collision: %q", version)
		}
		seen[version] = true
	}
}
