package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestSharedPilotDraftRequest(t *testing.T) {
	root := filepath.Join("..", "..", "bench", "jev", "testdata")
	qdata, err := os.ReadFile(filepath.Join(root, "shared-pilot.draft-v0.1.question.json"))
	if err != nil {
		t.Fatal(err)
	}
	var q question
	if decodeErr := json.Unmarshal(qdata, &q); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if validationErr := validateQuestion(q); validationErr != nil {
		t.Fatal(validationErr)
	}
	data, err := os.ReadFile(filepath.Join(root, "shared-pilot.draft-v0.1.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	cases, err := loadCases(data, q.Criteria, 6)
	if err != nil || len(cases) != 6 {
		t.Fatalf("cases=%d err=%v", len(cases), err)
	}
	labels := []string{"biology", "business", "chemistry", "computer science", "economics", "engineering", "health", "history", "law", "math", "other", "philosophy", "physics", "psychology"}
	if len(q.Criteria) != len(labels) {
		t.Fatal("expected 14 candidate labels")
	}
	// Deliberately disagree with the first fixture's expected biology label.
	// This is a synthetic response, not a Jev measurement.
	probs := make(map[string]float64, len(labels))
	for _, label := range labels {
		probs[label] = 0
	}
	probs["math"] = 1
	stub, err := json.Marshal(map[string]any{"model": "jev-1.13.0", "answers": map[string]any{"intent": map[string]any{"type": "choice", "choice": "math", "confidence": 0.6, "probabilities": probs}}})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, readErr := io.ReadAll(r.Body)
		if readErr != nil {
			t.Error(readErr)
		}
		var sent map[string]json.RawMessage
		if decodeErr := json.Unmarshal(body, &sent); decodeErr != nil {
			t.Error(decodeErr)
		}
		if len(sent) != 3 || sent["model"] == nil || sent["state"] == nil || sent["questions"] == nil {
			t.Error("unexpected request fields or leaked evaluation metadata")
		}
		want, _ := json.Marshal(request{Model: "jev-1.13.0", State: cases[0].State, Questions: map[string]question{"intent": q}})
		if !bytes.Equal(body, want) {
			t.Error("input/configuration changed or metadata leaked")
		}
		// Verify actual serialized candidate order, not Go map iteration order.
		pos := -1
		for _, label := range labels {
			next := strings.Index(string(body), `"`+label+`":`)
			if next <= pos {
				t.Errorf("candidate missing or out of order: %s", label)
			}
			pos = next
		}
		_, _ = w.Write(stub)
	}))
	defer server.Close()
	a, err := newAdapter(server.URL, "test-key", q, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer a.client.Close()
	var output bytes.Buffer
	if err := collect(context.Background(), a, cases[:1], runOptions{Timeout: time.Second}, "draft-data", "draft-question", json.NewEncoder(&output)); err != nil {
		t.Fatal(err)
	}
	var got record
	if err := json.Unmarshal(output.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.Valid || got.Correct == nil || *got.Correct || got.Expected == nil || *got.Expected != "biology" || got.RawResponse != string(stub) {
		t.Fatal("expected local-only scoring, a valid but wrong prediction, and unchanged response")
	}
}

func TestPilotDiagnosticScoring(t *testing.T) {
	root := filepath.Join("..", "..", "bench", "jev", "testdata")
	qdata, err := os.ReadFile(filepath.Join(root, "shared-pilot.draft-v0.1.question.json"))
	if err != nil {
		t.Fatal(err)
	}
	var q question
	if err := json.Unmarshal(qdata, &q); err != nil {
		t.Fatal(err)
	}
	load := func(name string) []testCase {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		cases, err := loadCases(data, q.Criteria, 6)
		if err != nil || len(cases) != 6 {
			t.Fatalf("cases=%d err=%v", len(cases), err)
		}
		return cases
	}
	original := load("shared-pilot.draft-v0.1.jsonl")
	diagnostic := load("shared-pilot.diagnostic-v0.1.jsonl")
	if original[5].Expected == nil || *original[5].Expected != "other" || diagnostic[5].Expected != nil {
		t.Fatal("expected original other label and diagnostic null label")
	}
	original[5].Expected = nil // Compare copies in memory, never rewrite the fixture.
	if !reflect.DeepEqual(original, diagnostic) {
		t.Fatal("diagnostic variant changed more than case 006's expected label")
	}
	for _, tc := range []struct {
		name  string
		mass  float64
		valid bool
	}{{"valid_but_unscored", 1, true}, {"invalid_still_rejected", 0.5, false}} {
		t.Run(tc.name, func(t *testing.T) {
			probs := make(map[string]float64, len(q.Criteria))
			for label := range q.Criteria {
				probs[label] = 0
			}
			// Physics is a synthetic observation, not a correct/incorrect verdict.
			probs["physics"] = tc.mass
			stub, err := json.Marshal(map[string]any{"model": "jev-1.13.0", "answers": map[string]any{"intent": map[string]any{"type": "choice", "choice": "physics", "confidence": 0.6, "probabilities": probs}}})
			if err != nil {
				t.Fatal(err)
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, readErr := io.ReadAll(r.Body)
				want, marshalErr := json.Marshal(request{Model: "jev-1.13.0", State: diagnostic[5].State, Questions: map[string]question{"intent": q}})
				if readErr != nil || marshalErr != nil || !bytes.Equal(body, want) {
					t.Error("diagnostic metadata changed the API request")
				}
				_, _ = w.Write(stub)
			}))
			defer server.Close()
			a, err := newAdapter(server.URL, "test-key", q, time.Second)
			if err != nil {
				t.Fatal(err)
			}
			defer a.client.Close()
			var output bytes.Buffer
			err = collect(context.Background(), a, diagnostic[5:], runOptions{Timeout: time.Second}, "mock-data", "mock-question", json.NewEncoder(&output))
			var got record
			if decodeErr := json.Unmarshal(output.Bytes(), &got); decodeErr != nil {
				t.Fatal(decodeErr)
			}
			if (err == nil) != tc.valid || got.Valid != tc.valid || got.Expected != nil || got.Correct != nil || got.Attempts != 1 || got.RawResponse != string(stub) {
				t.Fatalf("diagnostic scoring/validation mismatch: %+v err=%v", got, err)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(output.Bytes(), &fields); err != nil {
				t.Fatal(err)
			}
			if _, exists := fields["correct"]; exists {
				t.Fatal("diagnostic record must omit correctness, not encode false")
			}
			t.Logf("MOCK ONLY: case=%s expected=null prediction=physics probability_sum=%.1f contract_valid=%t correctness=omitted", got.ID, tc.mass, got.Valid)
		})
	}
}
