package services

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

// Byte offsets land past the end of the string for a code-point client.
func TestBuildPIIEntitiesReportsRuneOffsets(t *testing.T) {
	text := "こんにちは、私は John Smith です。電話は 415-555-0134"
	detections := []classification.PIIDetection{
		{EntityType: "PERSON", Text: "John Smith", Start: 25, End: 35, Confidence: 0.99},
		{EntityType: "PHONE_NUMBER", Text: "415-555-0134", Start: 55, End: 67, Confidence: 0.99},
	}

	entities := buildPIIEntities(text, detections, true, false, true, map[string]string{})
	runes := []rune(text)
	for _, e := range entities {
		if e.StartPos == nil || e.EndPos == nil {
			t.Fatalf("%s: positions were requested but not returned", e.Type)
		}
		if *e.EndPos > len(runes) {
			t.Fatalf("%s: end_position %d past the end of a %d-rune string", e.Type, *e.EndPos, len(runes))
		}
		if got := string(runes[*e.StartPos:*e.EndPos]); got != e.Value {
			t.Errorf("%s: slicing by the reported offsets gave %q, want %q", e.Type, got, e.Value)
		}
	}
}

func TestBuildPIIEntitiesLeavesAsciiOffsetsAlone(t *testing.T) {
	text := "Contact John Smith at 415-555-0134."
	detections := []classification.PIIDetection{
		{EntityType: "PERSON", Text: "John Smith", Start: 8, End: 18, Confidence: 0.9},
	}
	entities := buildPIIEntities(text, detections, true, false, true, map[string]string{})
	if *entities[0].StartPos != 8 || *entities[0].EndPos != 18 {
		t.Fatalf("got [%d,%d), want [8,18)", *entities[0].StartPos, *entities[0].EndPos)
	}
}

// omitempty on a plain int drops an offset of 0.
func TestBuildPIIEntitiesKeepsZeroStartPosition(t *testing.T) {
	text := "John Smith called yesterday."
	detections := []classification.PIIDetection{
		{EntityType: "PERSON", Text: "John Smith", Start: 0, End: 10, Confidence: 0.9},
	}
	entities := buildPIIEntities(text, detections, true, false, true, map[string]string{})
	if entities[0].StartPos == nil {
		t.Fatal("start_position missing for an entity at offset 0")
	}
	if *entities[0].StartPos != 0 {
		t.Fatalf("start_position = %d, want 0", *entities[0].StartPos)
	}
}
