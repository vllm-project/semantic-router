package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
)

const validResponse = `{"model":"jev-1.13.0","answers":{"intent":{"type":"choice","choice":"coding","confidence":0.6,"probabilities":{"writing":0.2,"coding":0.8}}},"usage":{"input_tokens":10,"output_tokens":2}}`

func fixtureQuestion() question {
	return question{Type: "choice", Instructions: "Classify the request", Criteria: map[string]string{"coding": "Code", "writing": "Prose"}}
}

func TestResponseContract(t *testing.T) {
	tests := []struct {
		name, body string
		valid      bool
	}{
		{"valid", validResponse, true},
		{"rounding_within_tolerance", strings.Replace(validResponse, `"coding":0.8`, `"coding":0.8005`, 1), true},
		{"empty", `{}`, false},
		{"malformed", `{`, false},
		{"unversioned", strings.ReplaceAll(validResponse, "jev-1.13.0", "jev-latest"), false},
		{"wrong_type", strings.Replace(validResponse, `"type":"choice"`, `"type":"score"`, 1), false},
		{"missing_label", strings.Replace(validResponse, `"writing":0.2,`, "", 1), false},
		{"extra_label", strings.Replace(validResponse, `"writing":0.2`, `"other":0.1,"writing":0.1`, 1), false},
		{"wrong_label_same_count", strings.Replace(validResponse, `"writing":0.2`, `"other":0.2`, 1), false},
		{"null_probability", strings.Replace(validResponse, `"writing":0.2`, `"writing":null`, 1), false},
		{"negative", strings.Replace(validResponse, `"writing":0.2`, `"writing":-0.2`, 1), false},
		{"over_one", strings.Replace(validResponse, `"coding":0.8`, `"coding":1.2`, 1), false},
		{"overflow", strings.Replace(validResponse, `"coding":0.8`, `"coding":1e999`, 1), false},
		{"sum_outside_tolerance", strings.Replace(validResponse, `"coding":0.8`, `"coding":0.81`, 1), false},
		{"confidence_cannot_fill_mass", strings.Replace(validResponse, `"coding":0.8`, `"coding":0.2`, 1), false},
		{"unknown_choice", strings.Replace(validResponse, `"choice":"coding"`, `"choice":"other"`, 1), false},
		{"wrong_argmax", strings.Replace(validResponse, `"choice":"coding"`, `"choice":"writing"`, 1), false},
		{"missing_confidence", strings.Replace(validResponse, `"confidence":0.6,`, "", 1), false},
		{"invalid_confidence", strings.Replace(validResponse, `"confidence":0.6`, `"confidence":1.1`, 1), false},
		{"null_confidence", strings.Replace(validResponse, `"confidence":0.6`, `"confidence":null`, 1), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, err := validateResponse([]byte(tt.body), "jev-1.13.0", fixtureQuestion().Criteria)
			if (err == nil) != tt.valid {
				t.Fatalf("valid=%v, err=%v", tt.valid, err)
			}
			if tt.name == "valid" && (*out.Answers["intent"].Probabilities["coding"] != 0.8 || *out.Answers["intent"].Confidence != 0.6) {
				t.Fatal("distribution or confidence was changed")
			}
		})
	}
}

func TestAdapterRequestAndResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/systemone" || r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("unexpected request method/path/auth")
		}
		var req request
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
		}
		if req.Model != "jev-1.13.0" || req.State != "debug this" || len(req.Questions) != 1 || req.Questions["intent"].Criteria["coding"] != "Code" {
			t.Errorf("unexpected request: %+v", req)
		}
		_, _ = w.Write([]byte(validResponse))
	}))
	defer server.Close()
	a, err := newAdapter(server.URL, "test-key", fixtureQuestion(), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer a.client.Close()
	_, wire, out, err := a.evaluate(context.Background(), "debug this")
	if err != nil || wire.Attempts != 1 || string(wire.Body) != validResponse || out == nil {
		t.Fatalf("wire=%+v err=%v", wire, err)
	}
}

func TestConnectorFailureRecords(t *testing.T) {
	for _, status := range []int{401, 429, 529, 503} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.WriteHeader(status)
				_, _ = w.Write([]byte(`{"error":"synthetic failure"}`))
			}))
			defer server.Close()
			a, err := newAdapter(server.URL, "test-key", fixtureQuestion(), time.Second)
			if err != nil {
				t.Fatal(err)
			}
			defer a.client.Close()
			var output bytes.Buffer
			cases := []testCase{{ID: "first", State: "hello"}, {ID: "second", State: "hello"}}
			err = collect(context.Background(), a, cases, runOptions{Timeout: time.Second}, "data", "question", json.NewEncoder(&output))
			if err == nil || calls.Load() != 1 {
				t.Fatalf("must stop after first failed call, calls=%d err=%v", calls.Load(), err)
			}
			var got record
			if err := json.Unmarshal(output.Bytes(), &got); err != nil {
				t.Fatal(err)
			}
			if got.Valid || got.Correct != nil || got.Status != status || got.ErrorKind != "status" || got.Attempts != 1 || got.RawResponse == "" {
				t.Fatalf("failure was not preserved: %+v", got)
			}
			if strings.Contains(output.String(), "test-key") || strings.Contains(output.String(), "Authorization") {
				t.Fatal("credential leaked")
			}
		})
	}
}

func TestCancellationAndDeadline(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(200 * time.Millisecond):
		}
	}))
	defer server.Close()
	a, err := newAdapter(server.URL, "test-key", fixtureQuestion(), 30*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	defer a.client.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, _, err = a.evaluate(ctx, "hello")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation, got %v", err)
	}
	_, _, _, err = a.evaluate(context.Background(), "hello")
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected deadline, got %v", err)
	}
}

func TestConnectorRejectsRedirectAndOversizedResponse(t *testing.T) {
	for _, mode := range []string{"redirect", "oversized", "malformed"} {
		t.Run(mode, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				switch mode {
				case "redirect":
					http.Redirect(w, r, "/untrusted", http.StatusTemporaryRedirect)
				case "oversized":
					_, _ = w.Write([]byte(strings.Repeat("x", (1<<20)+1)))
				case "malformed":
					_, _ = w.Write([]byte("not JSON"))
				}
			}))
			defer server.Close()
			a, err := newAdapter(server.URL, "test-key", fixtureQuestion(), time.Second)
			if err != nil {
				t.Fatal(err)
			}
			defer a.client.Close()
			_, _, _, err = a.evaluate(context.Background(), "hello")
			if err == nil || calls.Load() != 1 {
				t.Fatalf("calls=%d err=%v", calls.Load(), err)
			}
			if mode == "redirect" && !errors.Is(err, connector.ErrRedirectRejected) {
				t.Fatal(err)
			}
			if mode == "oversized" && !errors.Is(err, connector.ErrResponseTooLarge) {
				t.Fatal(err)
			}
		})
	}
}

func TestRecordSeparatesGroundTruthAndAmbiguity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if len(body) != 3 || body["expected"] != nil || body["group"] != nil {
			t.Error("evaluation metadata leaked into inference")
		}
		_, _ = w.Write([]byte(validResponse))
	}))
	defer server.Close()
	a, err := newAdapter(server.URL, "test-key", fixtureQuestion(), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer a.client.Close()
	var output bytes.Buffer
	err = collect(context.Background(), a, []testCase{{ID: "ambiguous", State: "help"}}, runOptions{Timeout: time.Second}, "d", "q", json.NewEncoder(&output))
	if err != nil {
		t.Fatal(err)
	}
	var got record
	if err := json.Unmarshal(output.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.Valid || got.Correct != nil || got.RawResponse != validResponse || got.Contract != contract {
		t.Fatalf("unexpected record: %+v", got)
	}
}

func TestLiveOptIn(t *testing.T) {
	if err := run(context.Background(), runOptions{}); err == nil || !strings.Contains(err.Error(), "--live") {
		t.Fatalf("expected explicit opt-in, got %v", err)
	}
}

func TestExistingOutputIsNeverOverwritten(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "test-key")
	dir := t.TempDir()
	input := filepath.Join(dir, "input.jsonl")
	qpath := filepath.Join(dir, "question.json")
	output := filepath.Join(dir, "output.jsonl")
	qdata, err := json.Marshal(fixtureQuestion())
	if err != nil {
		t.Fatal(err)
	}
	for path, data := range map[string][]byte{
		input:  []byte(`{"id":"one","group":"one","state":"debug this","expected":"coding"}`),
		qpath:  qdata,
		output: []byte("existing evidence"),
	} {
		if err = os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	err = run(context.Background(), runOptions{
		Inputs: input, Question: qpath, Output: output, Live: true,
		Location: "test", Revision: "test", Timeout: time.Second, MaxCases: 1,
	})
	if !errors.Is(err, os.ErrExist) {
		t.Fatalf("expected pre-network output refusal, got %v", err)
	}
	data, err := os.ReadFile(output)
	if err != nil || string(data) != "existing evidence" {
		t.Fatalf("output changed: %q, %v", data, err)
	}
}

func TestInvalidContractPreservesRawResponseAndStops(t *testing.T) {
	body := strings.Replace(validResponse, `"coding":0.8`, `"coding":0.3`, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()
	a, err := newAdapter(server.URL, "test-key", fixtureQuestion(), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer a.client.Close()
	var output bytes.Buffer
	err = collect(context.Background(), a, []testCase{{ID: "invalid", State: "hello"}}, runOptions{Timeout: time.Second}, "d", "q", json.NewEncoder(&output))
	if err == nil {
		t.Fatal("invalid distribution did not stop probe")
	}
	var got record
	if err = json.Unmarshal(output.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Valid || got.Correct != nil || got.RawResponse != body || got.Status != 200 || got.ErrorKind != "contract" {
		t.Fatalf("invalid evidence not preserved: %+v", got)
	}
}

func TestDevelopmentFixtures(t *testing.T) {
	// External Go tools run with the Router module as their working directory.
	root := filepath.Join("..", "..", "bench", "jev", "testdata")
	qdata, err := os.ReadFile(filepath.Join(root, "question.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	var q question
	if err = json.Unmarshal(qdata, &q); err != nil {
		t.Fatal(err)
	}
	if err = validateQuestion(q); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(root, "development.v1.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	cases, err := loadCases(data, q.Criteria, 12)
	if err != nil || len(cases) != 12 {
		t.Fatalf("cases=%d err=%v", len(cases), err)
	}
	if _, err := loadCases(data, q.Criteria, 11); err == nil {
		t.Fatal("budget limit not enforced")
	}
	if _, err := loadCases(append(data, data...), q.Criteria, 24); err == nil {
		t.Fatal("duplicate IDs not rejected")
	}
	scored := 0
	for _, c := range cases {
		if c.Expected != nil {
			scored++
		}
	}
	if scored != 10 {
		t.Fatalf("expected 10 scored cases, got %d", scored)
	}
}
