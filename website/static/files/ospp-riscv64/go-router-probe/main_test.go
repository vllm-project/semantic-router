package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// These fixtures catch accepting an invalid backend contract, including model echo.
func TestBackendContract(t *testing.T) {
	s := httptest.NewServer(backendHandler())
	defer s.Close()
	r, err := http.Get(s.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != 200 {
		t.Fatal(r.Status)
	}
	for _, model := range []string{"code-model", "general-model"} {
		c := newClient(1)
		if err := perform(c, s.URL, "backend", model); err != nil {
			t.Fatal(err)
		}
		c.CloseIdleConnections()
	}
	for _, body := range []string{`{"model":"bad","stream":false}`, `{"model":"code-model","stream":true}`, `{"model":"code-model","stream":false} {}`, `{`} {
		r, err := http.Post(s.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		r.Body.Close()
		if r.StatusCode != 400 {
			t.Fatalf("body %q: status %d", body, r.StatusCode)
		}
	}
	r, err = http.Post(s.URL+"/prefix/chat/completions", "application/json", strings.NewReader(`{"model":"code-model","stream":false}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	var v struct {
		Model   string
		Choices []struct {
			Message      struct{ Content string }
			FinishReason string `json:"finish_reason"`
		}
		Usage map[string]int
	}
	if err := json.NewDecoder(r.Body).Decode(&v); err != nil {
		t.Fatal(err)
	}
	if v.Model != "code-model" || len(v.Choices) != 1 || v.Choices[0].Message.Content != "ok" || v.Choices[0].FinishReason != "stop" || v.Usage["prompt_tokens"] != 1 || v.Usage["completion_tokens"] != 1 || v.Usage["total_tokens"] != 2 {
		t.Fatalf("bad contract: %+v", v)
	}
}

func TestBackendDefaultsOmittedStreamToBuffered(t *testing.T) {
	s := httptest.NewServer(backendHandler())
	defer s.Close()
	r, err := http.Post(s.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"code-model"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != 200 {
		t.Fatalf("omitted stream must default to non-streaming, got %d", r.StatusCode)
	}
}

// Real HTTP fixtures catch missing status, media type, model and content checks.
func TestResponseValidation(t *testing.T) {
	for _, tc := range []struct {
		name        string
		status      int
		media, body string
		ok          bool
	}{
		{"correct", 200, "application/json; charset=utf-8", `{"model":"code-model","choices":[{"message":{"content":"ok"}}]}`, true},
		{"wrong-model", 200, "application/json", `{"model":"general-model","choices":[{"message":{"content":"ok"}}]}`, false},
		{"non200", 503, "application/json", `{}`, false},
		{"media", 200, "text/plain", `{"model":"code-model","choices":[{"message":{"content":"ok"}}]}`, false},
		{"content", 200, "application/json", `{"model":"code-model","choices":[{"message":{"content":"wrong"}}]}`, false},
		{"malformed", 200, "application/json", `{`, false},
		{"trailing", 200, "application/json", `{"model":"code-model","choices":[{"message":{"content":"ok"}}]} {}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", tc.media)
				w.WriteHeader(tc.status)
				io.WriteString(w, tc.body)
			}))
			defer s.Close()
			c := newClient(1)
			defer c.CloseIdleConnections()
			err := perform(c, s.URL, "backend", "code-model")
			if (err == nil) != tc.ok {
				t.Fatalf("ok=%v, err=%v", tc.ok, err)
			}
		})
	}
}

// Catches sending the auto model directly or neglecting route-specific prompts.
func TestRouterPayloadAndKeepalive(t *testing.T) {
	var connections atomic.Int64
	var mu sync.Mutex
	var models, prompts []string
	s := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var q struct {
			Model    string
			Stream   bool
			Messages []struct{ Content string }
		}
		if err := json.NewDecoder(r.Body).Decode(&q); err != nil {
			t.Error(err)
		}
		mu.Lock()
		models = append(models, q.Model)
		if len(q.Messages) > 0 {
			prompts = append(prompts, q.Messages[0].Content)
		}
		mu.Unlock()
		if q.Stream || r.ProtoMajor != 1 {
			t.Error("streaming or non-HTTP/1 request")
		}
		model := "code-model"
		if len(prompts)%2 == 0 {
			model = "general-model"
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"model":%q,"choices":[{"message":{"content":"ok"}}]}`, model)
	}))
	s.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			connections.Add(1)
		}
	}
	s.Start()
	defer s.Close()
	c := newClient(1)
	defer c.CloseIdleConnections()
	for i, model := range []string{"code-model", "general-model", "code-model", "general-model"} {
		path := "backend"
		if i >= 2 {
			path = "router"
		}
		if err := perform(c, s.URL, path, model); err != nil {
			t.Fatal(err)
		}
	}
	if strings.Join(models, ",") != "code-model,general-model,vllm-sr/auto,vllm-sr/auto" {
		t.Fatal(models)
	}
	if len(prompts) != 4 || prompts[0] == prompts[1] || prompts[0] != prompts[2] {
		t.Fatal(prompts)
	}
	if connections.Load() != 1 {
		t.Fatalf("opened %d connections", connections.Load())
	}
}

// Catches counting warmups as measured requests or losing per-worker alternation.
func TestPhaseCounts(t *testing.T) {
	var calls atomic.Int64
	b := backendHandler()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls.Add(1); b.ServeHTTP(w, r) }))
	defer s.Close()
	p := runPhase(s.URL, "backend", 2, 1, 25*time.Millisecond)
	if !p.Passed || p.Requests < 2 || p.ErrorCount != 0 {
		t.Fatalf("%+v", p)
	}
	if calls.Load() != p.Requests+2 {
		t.Fatalf("calls=%d measured=%d", calls.Load(), p.Requests)
	}
	if p.RouteRequests["code-model"]+p.RouteRequests["general-model"] != p.Requests {
		t.Fatal(p.RouteRequests)
	}
	diff := p.RouteRequests["code-model"] - p.RouteRequests["general-model"]
	if diff < 0 || diff > 2 {
		t.Fatal(p.RouteRequests)
	}
	if p.StartedMonotonic <= 0 || p.FinishedMonotonic <= p.StartedMonotonic || p.Seconds < .025 || p.RequestsPerSecond <= 0 || p.P95MS < p.P50MS {
		t.Fatalf("%+v", p)
	}
	if delta := monotonicSeconds() - p.FinishedMonotonic; delta < 0 || delta > 2 {
		t.Fatal(delta)
	}
}

// Catches approximate quantiles and rounding errors at nearest-rank boundaries.
func TestHistogramNearestRank(t *testing.T) {
	h := histogram{}
	if h.quantile(.95) != 0 {
		t.Fatal("empty")
	}
	for _, d := range []time.Duration{1, 1, 2, 3, 100} {
		h.add(d * time.Microsecond)
	}
	if h.quantile(.5) != .002 || h.quantile(.95) != .1 {
		t.Fatalf("p50=%v p95=%v", h.quantile(.5), h.quantile(.95))
	}
	h.add(1900 * time.Nanosecond)
	if h.quantile(.5) != .001 {
		t.Fatal("microsecond truncation/rank")
	}
}

// Catches invalid values reaching setup, and accepting unsafe endpoint schemes.
func TestInvalidArguments(t *testing.T) {
	base := []string{"bench", "--endpoint", "http://127.0.0.1:8000", "--backend-endpoint", "http://127.0.0.1:8001", "--output", "/tmp/new.json"}
	for _, more := range [][]string{{"--seconds", "0"}, {"--seconds", "NaN"}, {"--seconds", "+Inf"}, {"--seconds", "1e30"}, {"--seconds", "1e-20"}, {"--repeats", "0"}, {"--concurrencies", "1,0"}, {"--concurrencies", ""}, {"--procs", "0"}, {"--endpoint", "https://localhost"}, {"--endpoint", "http://u:p@localhost"}, {"--endpoint", "http://localhost/?x=y"}, {"--output", "relative.json"}, {"extra"}} {
		if _, err := parseConfig(append(append([]string{}, base...), more...)); err == nil {
			t.Errorf("accepted %v", more)
		}
	}
	for _, args := range [][]string{nil, {"bogus"}, {"serve", "--procs", "0"}, {"serve", "--listen", "bad"}, {"bench"}} {
		if _, err := parseConfig(args); err == nil {
			t.Errorf("accepted %v", args)
		}
	}
	if _, err := parseConfig(base); err != nil {
		t.Fatal(err)
	}
}

// Catches overwrite, missing incremental snapshots, incorrect ordering and false success.
func TestBenchOutput(t *testing.T) {
	file := filepath.Join(t.TempDir(), "result.json")
	var observedPending atomic.Bool
	b := backendHandler()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, err := os.ReadFile(file)
		if err == nil {
			var v result
			if json.Unmarshal(data, &v) == nil && !v.Passed {
				observedPending.Store(true)
			}
		}
		var q map[string]json.RawMessage
		json.NewDecoder(r.Body).Decode(&q)
		if string(q["model"]) == `"vllm-sr/auto"` {
			model := "general-model"
			if strings.Contains(strings.ToLower(string(q["messages"])), "write code") {
				model = "code-model"
			}
			q["model"], _ = json.Marshal(model)
		}
		data, _ = json.Marshal(q)
		r.Body = io.NopCloser(strings.NewReader(string(data)))
		b.ServeHTTP(w, r)
	}))
	defer s.Close()
	cfg, err := parseConfig([]string{"bench", "--endpoint", s.URL, "--backend-endpoint", s.URL, "--seconds", "0.01", "--repeats", "2", "--concurrencies", "1,2", "--output", file})
	if err != nil {
		t.Fatal(err)
	}
	if err := runBench(cfg); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	var got result
	if err = json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if !got.Passed || len(got.Phases) != 8 || !observedPending.Load() {
		t.Fatalf("%+v pending=%v", got, observedPending.Load())
	}
	for i, want := range []string{"backend", "router", "backend", "router", "router", "backend", "router", "backend"} {
		if got.Phases[i].Path != want || !got.Phases[i].Passed {
			t.Fatalf("phase %d: %+v", i, got.Phases[i])
		}
	}
	if err := runBench(cfg); err == nil {
		t.Fatal("overwrote output")
	}
	after, _ := os.ReadFile(file)
	if string(after) != string(data) {
		t.Fatal("changed existing output")
	}
}

// Catches losing errors after the detail cap and continuing a failed run.
func TestFailedPhaseAndPersistence(t *testing.T) {
	var calls atomic.Int64
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) > 1 {
			http.Error(w, "unavailable", 503)
			return
		}
		backendHandler().ServeHTTP(w, r)
	}))
	defer s.Close()
	p := runPhase(s.URL, "backend", 1, 1, time.Second)
	if p.Passed || p.Requests != 1 || p.ErrorCount != 1 || len(p.Errors) != 1 {
		t.Fatalf("%+v", p)
	}
	file := filepath.Join(t.TempDir(), "failed.json")
	cfg, _ := parseConfig([]string{"bench", "--endpoint", s.URL, "--backend-endpoint", s.URL, "--output", file})
	if err := runBench(cfg); err == nil {
		t.Fatal("success on failed warmup")
	}
	data, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	var got result
	if json.Unmarshal(data, &got) != nil || got.Passed || got.Failure == "" || len(got.Phases) != 0 {
		t.Fatalf("%s", data)
	}
}

// Catches missing headers being invented as zeros, or malformed headers poisoning quantiles.
func TestRoutingHeaderHistogram(t *testing.T) {
	for _, tc := range []struct {
		name, header string
		present      bool
	}{
		{"numeric", "1.2345", true},
		{"zero", "0", true},
		{"absent", "", false},
		{"malformed", "broken", false},
		{"nan", "NaN", false},
		{"infinity", "+Inf", false},
		{"negative", "-1", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tc.header != "" {
					w.Header().Set("x-vsr-routing-latency-ms", tc.header)
				}
				w.Header().Set("Content-Type", "application/json")
				var q struct{ Messages []struct{ Content string } }
				json.NewDecoder(r.Body).Decode(&q)
				model := "general-model"
				if strings.Contains(strings.ToLower(q.Messages[0].Content), "write code") {
					model = "code-model"
				}
				fmt.Fprintf(w, `{"model":%q,"choices":[{"message":{"content":"ok"}}]}`, model)
			}))
			defer s.Close()
			p := runPhase(s.URL, "router", 1, 1, 5*time.Millisecond)
			if !p.Passed {
				t.Fatalf("%+v", p)
			}
			data, _ := json.Marshal(p)
			var v struct {
				Header *struct {
					Count, Missing int64
					P50            *float64 `json:"p50_ms"`
					P95            *float64 `json:"p95_ms"`
				} `json:"routing_header_ms"`
			}
			if err := json.Unmarshal(data, &v); err != nil {
				t.Fatal(err)
			}
			if v.Header == nil {
				t.Fatalf("missing header summary: %s", data)
			}
			h := v.Header
			if tc.present {
				want := 1.2345
				if tc.header == "0" {
					want = 0
				}
				if h.Count != p.Requests || h.Missing != 0 || h.P50 == nil || h.P95 == nil || *h.P50 != want || *h.P95 != want {
					t.Fatalf("%s", data)
				}
			} else if h.Count != 0 || h.Missing != p.Requests || h.P50 != nil || h.P95 != nil {
				t.Fatalf("%s", data)
			}
		})
	}
}

// The frozen smoke Router routes only the case-insensitive literal "write code".
// This fails if the code prompt drifts outside that fixture's routing rule.
func TestFrozenRouterKeywordRule(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var q struct {
			Model    string
			Messages []struct{ Content string }
		}
		if err := json.NewDecoder(r.Body).Decode(&q); err != nil {
			t.Error(err)
			http.Error(w, "bad JSON", 400)
			return
		}
		if q.Model != "vllm-sr/auto" || len(q.Messages) != 1 {
			t.Error("invalid Router request")
			http.Error(w, "bad request", 400)
			return
		}
		model := "general-model"
		if strings.Contains(strings.ToLower(q.Messages[0].Content), "write code") {
			model = "code-model"
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"model":%q,"choices":[{"message":{"content":"ok"}}]}`, model)
	}))
	defer s.Close()
	c := newClient(1)
	defer c.CloseIdleConnections()
	for _, model := range []string{"code-model", "general-model"} {
		if err := perform(c, s.URL, "router", model); err != nil {
			t.Fatal(err)
		}
	}
}
