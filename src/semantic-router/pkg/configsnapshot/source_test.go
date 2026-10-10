package configsnapshot

import (
	"context"
	"testing"
)

// pushedSource is a control plane that pushes documents and records the
// ACK or NACK of each.
type pushedSource struct {
	updates []Update
	acks    []uint64
	nacks   []Code
}

func (s *pushedSource) Run(ctx context.Context, apply func(context.Context, Update) (*Snapshot, error)) error {
	for _, update := range s.updates {
		snapshot, err := apply(ctx, update)
		if reasons := ReasonsOf(err); len(reasons) > 0 {
			s.nacks = append(s.nacks, reasons[0].Code)
			continue
		}
		if err != nil {
			return err
		}
		s.acks = append(s.acks, snapshot.Version())
	}
	return nil
}

func TestASourceFeedsTheLifecycleAndLearnsEachOutcome(t *testing.T) {
	manager := installed(t, &fakeRuntime{}, nil)
	rejected := documentUpdate("control-plane", "rejected")
	rejected.Config = nil
	source := &pushedSource{updates: []Update{
		documentUpdate("control-plane", "a"), rejected, documentUpdate("control-plane", "b"),
	}}
	if err := manager.Serve(context.Background(), source); err != nil {
		t.Fatal(err)
	}
	if len(source.acks) != 2 || source.acks[0] != 2 || source.acks[1] != 3 ||
		len(source.nacks) != 1 || source.nacks[0] != CodeInvalidDocument {
		t.Fatalf("acks %v, nacks %v; want versions 2 and 3 and one invalid_document", source.acks, source.nacks)
	}
	if manager.Active().Version() != 3 {
		t.Fatalf("active v%d", manager.Active().Version())
	}
}
