package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// Revalidate saved public/synthetic evidence without contacting the provider.
// This is not a new live benchmark or a guarantee about future API responses.
func TestPilotV01CapturedEvidence(t *testing.T) {
	root := filepath.Join("..", "..", "bench", "jev", "pilot-v0.1")
	read := func(name, hash string) []byte {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != hash {
			t.Fatalf("%s hash=%s, want %s", name, got, hash)
		}
		return data
	}
	const inputHash = "671c09f62dc9fc9b864efe54b0adfef0ec666f309f74b776dcec3d6d8cdd2ef6"
	const questionHash = "33f7ef7c4826df98a4bc23c2e0fc2302ea3ab61e18b8b45d0515e51e64f69738"
	var q question
	if err := json.Unmarshal(read("question.json", questionHash), &q); err != nil {
		t.Fatal(err)
	}
	if err := validateQuestion(q); err != nil {
		t.Fatal(err)
	}
	cases, err := loadCases(read("inputs.jsonl", inputHash), q.Criteria, 6)
	if err != nil {
		t.Fatal(err)
	}
	lines := bytes.Split(bytes.TrimSpace(read("jev-results.jsonl", "45a176c8b1e4524197601b0d7868bdeb44d568fe2b3d3732aa0192d46c19014d")), []byte("\n"))
	if len(cases) != 6 || len(lines) != len(cases) {
		t.Fatal("expected six cases and six records")
	}
	for i, line := range lines {
		var r record
		if err := json.Unmarshal(line, &r); err != nil {
			t.Fatal(err)
		}
		c := cases[i]
		want := request{Model: "jev-1.13.0", State: c.State, Questions: map[string]question{"intent": q}}
		if r.ID != c.ID || !reflect.DeepEqual(r.Expected, c.Expected) || !reflect.DeepEqual(r.Request, want) || r.DatasetSHA256 != inputHash || r.QuestionSHA256 != questionHash {
			t.Fatalf("%s: input/request/provenance mismatch", c.ID)
		}
		if _, err := validateResponse([]byte(r.RawResponse), "jev-1.13.0", q.Criteria); err != nil {
			t.Fatalf("%s: %v", c.ID, err)
		}
		if !r.Valid || r.Attempts != 1 || r.Status != 200 || r.Error != "" {
			t.Fatalf("%s: unexpected execution status", c.ID)
		}
		var raw struct {
			Answers map[string]struct {
				Choice string `json:"choice"`
			} `json:"answers"`
		}
		if err := json.Unmarshal([]byte(r.RawResponse), &raw); err != nil {
			t.Fatal(err)
		}
		if c.Expected == nil {
			if r.Correct != nil {
				t.Fatal("diagnostic case must not be scored")
			}
		} else if r.Correct == nil || *r.Correct != (raw.Answers["intent"].Choice == *c.Expected) {
			t.Fatalf("%s: scoring mismatch", c.ID)
		}
	}
}
