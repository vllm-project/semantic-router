package store

import (
	"context"
	"encoding/json"
	"testing"
)

func TestReplayUnmarkedConfidenceRemainsUnavailable(t *testing.T) {
	record := Record{ConfidenceScore: 1, JailbreakConfidence: 1, ResponseJailbreakConfidence: 1}
	raw, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]interface{}
	if decodeErr := json.Unmarshal(raw, &decoded); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if decoded["confidence_score"] != nil || decoded["confidence_score_available"] != false || decoded["jailbreak_confidence"] != nil || decoded["response_jailbreak_confidence"] != nil {
		t.Fatalf("legacy constants became model scores: %s", raw)
	}
	var restored Record
	if decodeErr := json.Unmarshal(raw, &restored); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if restored.ConfidenceScoreAvailable || restored.JailbreakScoreAvailable || restored.ResponseJailbreakScoreAvailable {
		t.Fatal("JSON roundtrip invented availability")
	}
}

func TestReplayReportedZeroSurvivesStores(t *testing.T) {
	input := Record{ID: "scored-zero", ConfidenceScoreAvailable: true, JailbreakScoreAvailable: true, ResponseJailbreakScoreAvailable: true, SignalErrorMatches: map[string]bool{"jailbreak:guard": true}}
	encoded, err := marshalPostgresSafety(input)
	if err != nil {
		t.Fatal(err)
	}
	var database Record
	if decodeErr := unmarshalPostgresSafety(encoded, &database); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if !database.ConfidenceScoreAvailable || !database.JailbreakScoreAvailable || !database.ResponseJailbreakScoreAvailable || !database.SignalErrorMatches["jailbreak:guard"] {
		t.Fatalf("Postgres metadata lost: %s", encoded)
	}
	memory := NewMemoryStore(10, 0)
	if _, addErr := memory.Add(context.Background(), input); addErr != nil {
		t.Fatal(addErr)
	}
	input.SignalErrorMatches["jailbreak:guard"] = false
	stored, found, err := memory.Get(context.Background(), input.ID)
	if err != nil || !found {
		t.Fatalf("get: %v %v", found, err)
	}
	if !stored.SignalErrorMatches["jailbreak:guard"] {
		t.Fatal("store retained caller's mutable map")
	}
	raw, err := json.Marshal(stored)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]interface{}
	if decodeErr := json.Unmarshal(raw, &decoded); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	for _, key := range []string{"confidence_score", "jailbreak_confidence", "response_jailbreak_confidence"} {
		if decoded[key] != float64(0) {
			t.Fatalf("actual zero lost for %s: %s", key, raw)
		}
	}
	var legacy Record
	if decodeErr := unmarshalPostgresSafety([]byte(`{"jailbreak_confidence":1}`), &legacy); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if legacy.JailbreakScoreAvailable || legacy.ConfidenceScoreAvailable {
		t.Fatal("unmarked Postgres legacy score became available")
	}
}
