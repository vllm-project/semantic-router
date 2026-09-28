// Package soak implements a long-running load + memory-timeseries harness for a
// running semantic-router stack (Envoy gateway -> router extproc -> backend).
// It is deliberately a standalone binary rather than an e2e testcase: soak runs
// are reporting-only and must not gate the e2e suite.
package soak

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	metricRSS       = "process_resident_memory_bytes"
	metricGoroutine = "go_goroutines"
	metricSys       = "go_memstats_sys_bytes"
	metricReleased  = "go_memstats_heap_released_bytes"
	metricHeapAlloc = "go_memstats_heap_alloc_bytes"
	metricHeapInuse = "go_memstats_heap_inuse_bytes"
)

// Sample is one scrape of the router's memory surface.
type Sample struct {
	TS         time.Time `json:"ts"`
	Phase      string    `json:"phase"`
	Round      int       `json:"round"`
	RSS        float64   `json:"rss_bytes"`
	Goroutines float64   `json:"goroutines"`
	Sys        float64   `json:"sys_bytes"`
	Released   float64   `json:"released_bytes"`
	HeapAlloc  float64   `json:"heap_alloc_bytes"`
	HeapInuse  float64   `json:"heap_inuse_bytes"`
	// GoTotal is the address space the Go runtime still owns, and NativeApprox
	// the remainder of RSS: cgo/model allocations, thread stacks, mmap'd files.
	GoTotal      float64 `json:"go_total_bytes"`
	NativeApprox float64 `json:"native_approx_bytes"`
}

// SamplerConfig configures periodic scraping and watermark-triggered dumps.
type SamplerConfig struct {
	MetricsURL string
	PprofURL   string
	OutDir     string
	Interval   time.Duration
	RouterPID  int
	// WatermarkGrowth is the fractional RSS increase over the previous high
	// water mark required to trigger a heap dump.
	WatermarkGrowth float64
}

// Sampler scrapes /metrics on a fixed cadence, streams every sample to disk and
// captures heap profiles when RSS sets a new high water mark.
type Sampler struct {
	cfg    SamplerConfig
	client *http.Client

	mu      sync.Mutex
	samples []Sample
	phase   string
	round   int
	highRSS float64
	errs    int

	tsFile  *os.File
	dumping atomic.Bool
	dumpWG  sync.WaitGroup
	done    chan struct{}
}

// NewSampler prepares the output tree and opens the timeseries stream.
func NewSampler(cfg SamplerConfig) (*Sampler, error) {
	if cfg.Interval <= 0 {
		cfg.Interval = 5 * time.Second
	}
	if cfg.WatermarkGrowth <= 0 {
		cfg.WatermarkGrowth = 0.02
	}
	if err := os.MkdirAll(filepath.Join(cfg.OutDir, "profiles"), 0o755); err != nil {
		return nil, fmt.Errorf("create profile dir: %w", err)
	}
	f, err := os.Create(filepath.Join(cfg.OutDir, "timeseries.json"))
	if err != nil {
		return nil, fmt.Errorf("create timeseries file: %w", err)
	}
	return &Sampler{
		cfg:     cfg,
		client:  &http.Client{},
		phase:   "init",
		tsFile:  f,
		done:    make(chan struct{}),
		samples: make([]Sample, 0, 4096),
	}, nil
}

// SetPhase labels every subsequent sample.
func (s *Sampler) SetPhase(phase string, round int) {
	s.mu.Lock()
	s.phase, s.round = phase, round
	s.mu.Unlock()
}

// Start runs the scrape loop until ctx is cancelled.
func (s *Sampler) Start(ctx context.Context) {
	go func() {
		defer close(s.done)
		ticker := time.NewTicker(s.cfg.Interval)
		defer ticker.Stop()
		s.collect(ctx)
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				s.collect(ctx)
			}
		}
	}()
}

// Close waits for the loop and any in-flight dump, then closes the stream.
func (s *Sampler) Close() error {
	<-s.done
	s.dumpWG.Wait()
	return s.tsFile.Close()
}

func (s *Sampler) collect(ctx context.Context) {
	body, err := s.fetch(ctx, s.cfg.MetricsURL, 10*time.Second)
	if err != nil {
		s.mu.Lock()
		s.errs++
		s.mu.Unlock()
		return
	}
	m := ParseMetrics(body)

	s.mu.Lock()
	sample := Sample{
		TS:         time.Now().UTC(),
		Phase:      s.phase,
		Round:      s.round,
		RSS:        m[metricRSS],
		Goroutines: m[metricGoroutine],
		Sys:        m[metricSys],
		Released:   m[metricReleased],
		HeapAlloc:  m[metricHeapAlloc],
		HeapInuse:  m[metricHeapInuse],
	}
	sample.GoTotal = sample.Sys - sample.Released
	sample.NativeApprox = sample.RSS - sample.GoTotal
	s.samples = append(s.samples, sample)
	_ = json.NewEncoder(s.tsFile).Encode(sample)

	newHigh := sample.RSS > s.highRSS*(1+s.cfg.WatermarkGrowth)
	if newHigh {
		s.highRSS = sample.RSS
	}
	phase := s.phase
	s.mu.Unlock()

	if newHigh {
		s.dumpAsync(fmt.Sprintf("watermark-%s", phase))
	}
}

func (s *Sampler) dumpAsync(label string) {
	if !s.dumping.CompareAndSwap(false, true) {
		return
	}
	s.dumpWG.Go(func() {
		defer s.dumping.Store(false)
		_ = s.Dump(context.Background(), label)
	})
}

// Dump writes a heap profile (and on linux the router's smaps_rollup) tagged
// with label.
func (s *Sampler) Dump(ctx context.Context, label string) error {
	stamp := time.Now().UTC().Format("20060102-150405")
	name := fmt.Sprintf("heap-%s-%s.pb.gz", sanitizeLabel(label), stamp)
	body, err := s.fetch(ctx, strings.TrimSuffix(s.cfg.PprofURL, "/")+"/debug/pprof/heap", 90*time.Second)
	if err != nil {
		return fmt.Errorf("fetch heap profile: %w", err)
	}
	if err := os.WriteFile(filepath.Join(s.cfg.OutDir, "profiles", name), body, 0o644); err != nil {
		return fmt.Errorf("write heap profile: %w", err)
	}
	if runtime.GOOS == "linux" && s.cfg.RouterPID > 0 {
		raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/smaps_rollup", s.cfg.RouterPID))
		if err == nil {
			smaps := fmt.Sprintf("smaps-%s-%s.txt", sanitizeLabel(label), stamp)
			_ = os.WriteFile(filepath.Join(s.cfg.OutDir, "profiles", smaps), raw, 0o644)
		}
	}
	return nil
}

func (s *Sampler) fetch(ctx context.Context, url string, timeout time.Duration) ([]byte, error) {
	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil, fmt.Errorf("unexpected status %d from %s", resp.StatusCode, url)
	}
	return io.ReadAll(resp.Body)
}

// Samples returns a snapshot of everything collected so far.
func (s *Sampler) Samples() []Sample {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Sample, len(s.samples))
	copy(out, s.samples)
	return out
}

// ScrapeErrors reports how many scrapes failed; a non-zero count weakens every
// downstream conclusion and is surfaced in the summary.
func (s *Sampler) ScrapeErrors() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.errs
}

// ParseMetrics extracts the metrics this harness cares about from a prometheus
// text exposition body.
func ParseMetrics(body []byte) map[string]float64 {
	wanted := map[string]bool{
		metricRSS: true, metricGoroutine: true, metricSys: true,
		metricReleased: true, metricHeapAlloc: true, metricHeapInuse: true,
	}
	out := make(map[string]float64, len(wanted))
	for line := range strings.SplitSeq(string(body), "\n") {
		if line == "" || line[0] == '#' {
			continue
		}
		idx := strings.IndexByte(line, ' ')
		if idx <= 0 {
			continue
		}
		name := line[:idx]
		if !wanted[name] {
			continue
		}
		v, err := strconv.ParseFloat(strings.TrimSpace(line[idx+1:]), 64)
		if err != nil {
			continue
		}
		out[name] = v
	}
	return out
}

func sanitizeLabel(label string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			return r
		default:
			return '-'
		}
	}, label)
}
