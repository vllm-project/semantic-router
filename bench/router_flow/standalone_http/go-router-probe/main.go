// go-router-probe is a Linux-only, dependency-free diagnostic HTTP/1.1 load tool.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"mime"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

const maxErrorDetails = 32

type config struct {
	Command         string  `json:"command"`
	Listen          string  `json:"listen,omitempty"`
	Endpoint        string  `json:"endpoint,omitempty"`
	BackendEndpoint string  `json:"backend_endpoint,omitempty"`
	Seconds         float64 `json:"seconds"`
	Repeats         int     `json:"repeats"`
	Concurrencies   []int   `json:"concurrencies"`
	Procs           int     `json:"procs"`
	Output          string  `json:"output,omitempty"`
}

func parseConfig(args []string) (config, error) {
	c := config{Procs: 2, Seconds: 15, Repeats: 3}
	if len(args) == 0 {
		return c, errors.New("expected serve or bench")
	}
	c.Command = args[0]
	f := flag.NewFlagSet(c.Command, flag.ContinueOnError)
	f.SetOutput(io.Discard)
	f.IntVar(&c.Procs, "procs", 2, "GOMAXPROCS")
	concurrencies := "1,4,16,64"
	switch c.Command {
	case "serve":
		f.StringVar(&c.Listen, "listen", "127.0.0.1:9000", "listen address")
	case "bench":
		f.StringVar(&c.Endpoint, "endpoint", "", "router base URL or full chat/completions URL")
		f.StringVar(&c.BackendEndpoint, "backend-endpoint", "", "backend base URL or full chat/completions URL")
		f.Float64Var(&c.Seconds, "seconds", 15, "measurement duration per phase")
		f.IntVar(&c.Repeats, "repeats", 3, "repeat count")
		f.StringVar(&concurrencies, "concurrencies", concurrencies, "comma-separated positive worker counts")
		f.StringVar(&c.Output, "output", "", "fresh absolute JSON output path")
	default:
		return c, fmt.Errorf("unknown command %q", c.Command)
	}
	if err := f.Parse(args[1:]); err != nil {
		return c, err
	}
	if f.NArg() != 0 || c.Procs <= 0 {
		return c, errors.New("unexpected arguments or nonpositive procs")
	}
	if c.Command == "serve" {
		host, port, err := net.SplitHostPort(c.Listen)
		p, e := strconv.Atoi(port)
		if err != nil || e != nil || host == "" || p < 0 || p > 65535 {
			return c, errors.New("invalid listen address")
		}
		return c, nil
	}
	if math.IsNaN(c.Seconds) || math.IsInf(c.Seconds, 0) || c.Seconds < 1e-9 || c.Seconds >= float64(math.MaxInt64)/1e9 || c.Repeats <= 0 {
		return c, errors.New("seconds must be finite, positive and representable as a duration; repeats must be positive")
	}
	for _, v := range strings.Split(concurrencies, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil || n <= 0 {
			return c, errors.New("concurrencies must be positive integers")
		}
		c.Concurrencies = append(c.Concurrencies, n)
	}
	for _, endpoint := range []string{c.Endpoint, c.BackendEndpoint} {
		u, err := url.Parse(endpoint)
		if err != nil || u.Scheme != "http" || u.Hostname() == "" || u.User != nil || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" {
			return c, fmt.Errorf("expected plain HTTP endpoint without credentials, query or fragment: %q", endpoint)
		}
		if port := u.Port(); port != "" {
			p, e := strconv.Atoi(port)
			if e != nil || p < 1 || p > 65535 {
				return c, errors.New("invalid endpoint port")
			}
		}
	}
	if !filepath.IsAbs(c.Output) {
		return c, errors.New("output must be an absolute fresh path")
	}
	return c, nil
}

func backendHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			http.NotFound(w, r)
			return
		}
		var q struct {
			Model  string `json:"model"`
			Stream *bool  `json:"stream"`
		}
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
		if err := decoder.Decode(&q); err != nil {
			http.Error(w, "invalid JSON", 400)
			return
		}
		var extra any
		if decoder.Decode(&extra) != io.EOF || (q.Model != "code-model" && q.Model != "general-model") || (q.Stream != nil && *q.Stream) {
			http.Error(w, "require code-model/general-model and a non-streaming request", 400)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id":"probe-fixed","object":"chat.completion","created":0,"model":%q,"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`, q.Model)
	})
}

func newClient(concurrency int) *http.Client {
	return &http.Client{
		Timeout:       10 * time.Second,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse },
		Transport: &http.Transport{
			Proxy: nil, ForceAttemptHTTP2: false,
			DialContext:     (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
			MaxConnsPerHost: concurrency, MaxIdleConnsPerHost: concurrency, MaxIdleConns: concurrency,
			IdleConnTimeout: 90 * time.Second, ResponseHeaderTimeout: 10 * time.Second,
		},
	}
}

// Bodies are prepared once: load generation does not marshal JSON per request.
var requestBodies = func() map[string][]byte {
	bodies := make(map[string][]byte)
	for _, path := range []string{"backend", "router"} {
		for _, model := range []string{"code-model", "general-model"} {
			wireModel := model
			if path == "router" {
				wireModel = "vllm-sr/auto"
			}
			prompt := "Explain photosynthesis"
			if model == "code-model" {
				prompt = "write code for binary search"
			}
			b, _ := json.Marshal(map[string]any{"model": wireModel, "stream": false, "messages": []map[string]string{{"role": "user", "content": prompt}}, "max_tokens": 1})
			bodies[path+"/"+model] = b
		}
	}
	return bodies
}()

func completionURL(endpoint string) string {
	u, _ := url.Parse(endpoint)
	u.Path = strings.TrimRight(u.Path, "/")
	if !strings.HasSuffix(u.Path, "/chat/completions") {
		if !strings.HasSuffix(u.Path, "/v1") {
			u.Path += "/v1"
		}
		u.Path += "/chat/completions"
	}
	u.RawPath = ""
	return u.String()
}

func perform(c *http.Client, endpoint, path, model string) error {
	return performObserved(c, endpoint, path, model, nil)
}

func performObserved(c *http.Client, endpoint, path, model string, observe func(string)) error {
	req, err := http.NewRequest(http.MethodPost, completionURL(endpoint), bytes.NewReader(requestBodies[path+"/"+model]))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	// Bounded reads prevent a misconfigured router from exhausting client memory.
	data, err := io.ReadAll(io.LimitReader(resp.Body, (1<<20)+1))
	if err != nil {
		return err
	}
	if len(data) > 1<<20 {
		return errors.New("response exceeds 1 MiB")
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP status %d", resp.StatusCode)
	}
	media, _, err := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if err != nil || media != "application/json" {
		return errors.New("expected application/json response")
	}
	var v struct {
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &v); err != nil {
		return fmt.Errorf("response JSON: %w", err)
	}
	if v.Model != model {
		return fmt.Errorf("expected model %q, got %q", model, v.Model)
	}
	if len(v.Choices) != 1 || v.Choices[0].Message.Content != "ok" {
		return errors.New("expected one choice with message.content=ok")
	}
	if observe != nil {
		observe(resp.Header.Get("x-vsr-routing-latency-ms"))
	}
	return nil
}

// Sparse integer-microsecond buckets contain counts, never raw samples. Latencies
// truncate to 1 us resolution; quantiles use nearest rank ceil(q*N), in ms.
type histogram map[int64]int64

func (h histogram) add(d time.Duration) { h[d.Microseconds()]++ }
func (h histogram) quantile(q float64) float64 {
	var total int64
	keys := make([]int64, 0, len(h))
	for k, n := range h {
		keys = append(keys, k)
		total += n
	}
	if total == 0 {
		return 0
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	rank := int64(math.Ceil(q * float64(total)))
	var seen int64
	for _, k := range keys {
		seen += h[k]
		if seen >= rank {
			return float64(k) / 1000
		}
	}
	return 0
}

// CLOCK_MONOTONIC (not a process-relative clock) matches Python time.monotonic
// on Linux. SYS_CLOCK_GETTIME is supplied by Go for both amd64 and riscv64.
func monotonicSeconds() float64 {
	var ts syscall.Timespec
	_, _, errno := syscall.Syscall(syscall.SYS_CLOCK_GETTIME, 1, uintptr(unsafe.Pointer(&ts)), 0)
	if errno != 0 {
		panic(errno)
	}
	return float64(ts.Sec) + float64(ts.Nsec)/1e9
}

type phase struct {
	Path              string           `json:"path"`
	Repeat            int              `json:"repeat"`
	Concurrency       int              `json:"concurrency"`
	StartedMonotonic  float64          `json:"started_monotonic"`
	FinishedMonotonic float64          `json:"finished_monotonic"`
	Seconds           float64          `json:"seconds"`
	Requests          int64            `json:"requests"`
	ErrorCount        int64            `json:"error_count"`
	Errors            []string         `json:"errors"`
	RouteRequests     map[string]int64 `json:"route_requests"`
	RequestsPerSecond float64          `json:"requests_per_second"`
	P50MS             float64          `json:"p50_ms"`
	P95MS             float64          `json:"p95_ms"`
	Passed            bool             `json:"passed"`
	RoutingHeaderMS   *routingHeader   `json:"routing_header_ms,omitempty"`
}

// Header values already have the Router's own ms resolution. Keep a sparse
// frequency histogram of those numeric values without rounding or raw samples.
// Missing includes absent, malformed, negative and nonfinite values. Only
// successful validated routed responses contribute. Empty quantiles are null.
type routingHeader struct {
	Count   int64    `json:"count"`
	Missing int64    `json:"missing"`
	P50MS   *float64 `json:"p50_ms"`
	P95MS   *float64 `json:"p95_ms"`
}

type headerHistogram map[float64]int64

func (h headerHistogram) summary(missing int64) *routingHeader {
	r := &routingHeader{Missing: missing}
	keys := make([]float64, 0, len(h))
	for v, n := range h {
		r.Count += n
		keys = append(keys, v)
	}
	if r.Count == 0 {
		return r
	}
	sort.Float64s(keys)
	quantile := func(q float64) *float64 {
		rank := int64(math.Ceil(q * float64(r.Count)))
		var seen int64
		for _, v := range keys {
			seen += h[v]
			if seen >= rank {
				value := v
				return &value
			}
		}
		return nil
	}
	r.P50MS = quantile(.5)
	r.P95MS = quantile(.95)
	return r
}

type workerResult struct {
	requests      int64
	errorCount    int64
	errors        []string
	routes        [2]int64
	latencies     histogram
	headers       headerHistogram
	headerMissing int64
}

// A best-effort request per worker warms persistent connections before the start
// barrier. No new requests start after the deadline or first error; in-flight
// requests drain (at most the 10 s request timeout) and count in elapsed time.
func runPhase(endpoint, path string, concurrency, repeat int, duration time.Duration) phase {
	p := phase{Path: path, Repeat: repeat, Concurrency: concurrency, Errors: []string{}, RouteRequests: map[string]int64{"code-model": 0, "general-model": 0}}
	c := newClient(concurrency)
	defer c.CloseIdleConnections()
	start := make(chan struct{})
	ready := make(chan error, concurrency)
	results := make(chan workerResult, concurrency)
	var stopped atomic.Bool
	var deadline time.Time
	for w := 0; w < concurrency; w++ {
		go func() {
			r := workerResult{latencies: histogram{}, headers: headerHistogram{}}
			err := perform(c, endpoint, path, "code-model")
			ready <- err
			<-start
			models := [2]string{"code-model", "general-model"}
			for i := 0; !stopped.Load() && time.Now().Before(deadline); i++ {
				index := i % 2
				before := time.Now()
				err := performObserved(c, endpoint, path, models[index], func(value string) {
					if path != "router" {
						return
					}
					ms, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
					if err != nil || math.IsNaN(ms) || math.IsInf(ms, 0) || ms < 0 {
						r.headerMissing++
						return
					}
					r.headers[ms]++
				})
				r.latencies.add(time.Since(before))
				r.requests++
				r.routes[index]++
				if err != nil {
					r.errorCount++
					if len(r.errors) < maxErrorDetails {
						r.errors = append(r.errors, err.Error())
					}
					stopped.Store(true)
				}
			}
			results <- r
		}()
	}
	for w := 0; w < concurrency; w++ {
		if err := <-ready; err != nil {
			p.ErrorCount++
			if len(p.Errors) < maxErrorDetails {
				p.Errors = append(p.Errors, "worker warmup: "+err.Error())
			}
		}
	}
	if p.ErrorCount > 0 {
		stopped.Store(true)
	}
	p.StartedMonotonic = monotonicSeconds()
	deadline = time.Now().Add(duration)
	close(start)
	h := histogram{}
	headers := headerHistogram{}
	var headerMissing int64
	for w := 0; w < concurrency; w++ {
		r := <-results
		p.Requests += r.requests
		p.ErrorCount += r.errorCount
		p.RouteRequests["code-model"] += r.routes[0]
		p.RouteRequests["general-model"] += r.routes[1]
		for _, err := range r.errors {
			if len(p.Errors) < maxErrorDetails {
				p.Errors = append(p.Errors, err)
			}
		}
		for k, n := range r.latencies {
			h[k] += n
		}
		for k, n := range r.headers {
			headers[k] += n
		}
		headerMissing += r.headerMissing
	}
	p.FinishedMonotonic = monotonicSeconds()
	p.Seconds = p.FinishedMonotonic - p.StartedMonotonic
	if p.Seconds > 0 {
		p.RequestsPerSecond = float64(p.Requests) / p.Seconds
	}
	p.P50MS = h.quantile(.50)
	p.P95MS = h.quantile(.95)
	p.Passed = p.ErrorCount == 0 && p.Requests > 0
	if path == "router" {
		p.RoutingHeaderMS = headers.summary(headerMissing)
	}
	return p
}

type result struct {
	Config               config  `json:"config"`
	Passed               bool    `json:"passed"`
	Failure              string  `json:"failure,omitempty"`
	LatencyResolutionUS  int     `json:"latency_resolution_us"`
	Quantiles            string  `json:"quantiles"`
	RequestAccounting    string  `json:"request_accounting"`
	RoutingHeaderMeaning string  `json:"routing_header_meaning"`
	Phases               []phase `json:"phases"`
}

// Write a complete snapshot to a same-directory temporary file, then publish
// atomically. Link provides no-clobber creation; subsequent snapshots use rename.
func writeResult(path string, v result, fresh bool) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".go-router-probe-*.json")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err = f.Write(append(data, '\n')); err != nil {
		f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	if fresh {
		return os.Link(f.Name(), path)
	}
	return os.Rename(f.Name(), path)
}

func runBench(c config) error {
	r := result{Config: c, LatencyResolutionUS: 1, Quantiles: "nearest-rank; client latencies truncated to integer microseconds; routing header retains emitted numeric ms precision", RequestAccounting: "attempts including errors; route_requests by expected model; elapsed includes in-flight drain; warmups excluded", RoutingHeaderMeaning: "Router internal routing-processing interval before forwarding, not total server handling; successful routed responses only; missing includes unusable values; empty quantiles are null", Phases: []phase{}}
	if err := writeResult(c.Output, r, true); err != nil {
		return fmt.Errorf("create fresh output: %w", err)
	}
	fail := func(err error) error {
		r.Passed = false
		r.Failure = err.Error()
		if saveErr := writeResult(c.Output, r, false); saveErr != nil {
			return errors.Join(err, saveErr)
		}
		return err
	}
	for _, path := range []string{"backend", "router"} {
		endpoint := c.BackendEndpoint
		if path == "router" {
			endpoint = c.Endpoint
		}
		client := newClient(1)
		for i := 0; i < 20; i++ {
			model := "code-model"
			if i%2 == 1 {
				model = "general-model"
			}
			if err := perform(client, endpoint, path, model); err != nil {
				client.CloseIdleConnections()
				return fail(fmt.Errorf("%s warmup %d: %w", path, i+1, err))
			}
		}
		client.CloseIdleConnections()
	}
	for repeat := 1; repeat <= c.Repeats; repeat++ {
		for _, concurrency := range c.Concurrencies {
			paths := []string{"backend", "router"}
			if repeat%2 == 0 {
				paths = []string{"router", "backend"}
			}
			for _, path := range paths {
				endpoint := c.BackendEndpoint
				if path == "router" {
					endpoint = c.Endpoint
				}
				p := runPhase(endpoint, path, concurrency, repeat, time.Duration(c.Seconds*1e9))
				r.Phases = append(r.Phases, p)
				if !p.Passed {
					return fail(fmt.Errorf("%s repeat %d concurrency %d failed (%d errors)", path, repeat, concurrency, p.ErrorCount))
				}
				if err := writeResult(c.Output, r, false); err != nil {
					return err
				}
			}
		}
	}
	r.Passed = true
	return writeResult(c.Output, r, false)
}

func main() {
	c, err := parseConfig(os.Args[1:])
	if err == nil {
		runtime.GOMAXPROCS(c.Procs)
		if c.Command == "serve" {
			s := &http.Server{Addr: c.Listen, Handler: backendHandler(), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 15 * time.Second, WriteTimeout: 15 * time.Second, IdleTimeout: 90 * time.Second, MaxHeaderBytes: 1 << 20}
			err = s.ListenAndServe()
		} else {
			err = runBench(c)
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
