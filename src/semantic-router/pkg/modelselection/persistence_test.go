package modelselection

import (
	"reflect"
	"testing"
)

func TestTrainingRecordArtifactRoundTrip(t *testing.T) {
	want := TrainingRecord{
		QueryEmbedding:    []float64{0.25, 0.75},
		SelectedModel:     "model-a",
		ResponseLatencyNs: 123456,
		ResponseQuality:   0.9,
		Success:           true,
		TimestampUnix:     123456789,
	}
	if got := fromSerializable(toSerializable(want)); !reflect.DeepEqual(got, want) {
		t.Fatalf("artifact record = %+v, want %+v", got, want)
	}
}
