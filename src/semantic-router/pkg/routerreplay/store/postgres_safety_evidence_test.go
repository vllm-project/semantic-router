package store

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestPostgresSafetyMetadataRoundTripAndLegacyAvailability(t *testing.T) {
	input := Record{JailbreakDetected: true, JailbreakType: "jailbreak", JailbreakDecision: &tasks.LabelDecision{Label: "jailbreak", SourceLabel: "unsafe"}, HallucinationScoreAvailable: true, HallucinationScoreKind: "max_hallucinated_token_score"}
	insert, err := newPostgresInsertRecord(input)
	if err != nil {
		t.Fatal(err)
	}
	idx := columnIndex(t, "safety_evidence")
	data, ok := insert.args()[idx].([]byte)
	if !ok {
		t.Fatal("missing safety JSON insert")
	}
	row := postgresRecordRow{signalsJSON: []byte("{}"), safetyEvidenceJSON: data}
	if row.scanDestinations()[idx] != &row.safetyEvidenceJSON {
		t.Fatal("safety column scan misaligned")
	}
	output, err := row.decode()
	if err != nil {
		t.Fatal(err)
	}
	if !output.JailbreakDetected || output.JailbreakDecision == nil || output.JailbreakDecision.SourceLabel != "unsafe" || !output.HallucinationScoreAvailable || output.HallucinationScoreKind != input.HallucinationScoreKind {
		t.Fatalf("safety metadata lost: %+v", output)
	}
	legacy := postgresRecordRow{signalsJSON: []byte("{}"), record: Record{HallucinationConfidence: .9}}
	previous, err := legacy.decode()
	if err != nil || previous.HallucinationScoreAvailable {
		t.Fatalf("legacy numeric field became a declared score: %+v %v", previous, err)
	}
	if !strings.Contains(postgresCreateTableQuery("router_replay_records"), "ADD COLUMN IF NOT EXISTS safety_evidence JSONB") {
		t.Fatal("existing database cannot migrate safety metadata")
	}
}

func TestMemoryHallucinationStatusRetainsScoreAvailability(t *testing.T) {
	memory := NewMemoryStore(10, 0)
	id, err := memory.Add(context.Background(), Record{ID: "score-availability"})
	if err != nil {
		t.Fatal(err)
	}
	if updateErr := memory.UpdateHallucinationStatus(context.Background(), id, true, 0, []string{"span"}, nil, HallucinationScore{Available: false}); updateErr != nil {
		t.Fatal(updateErr)
	}
	record, found, err := memory.Get(context.Background(), id)
	if err != nil || !found || record.HallucinationScoreAvailable || !record.HallucinationDetected {
		t.Fatalf("record=%+v err=%v", record, err)
	}
	if updateErr := memory.UpdateHallucinationStatus(context.Background(), id, true, 0, []string{"span"}, nil, HallucinationScore{Available: true, Kind: "probability"}); updateErr != nil {
		t.Fatal(updateErr)
	}
	record, _, _ = memory.Get(context.Background(), id)
	if !record.HallucinationScoreAvailable || record.HallucinationScoreKind != "probability" {
		t.Fatal("actual reported zero was lost")
	}
}
