package soak

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/performance"
)

const lowSuccessRateFloor = 0.90

// Config is the user-facing configuration, one field per flag.
type Config struct {
	GatewayURL     string        `json:"gateway_url"`
	MetricsURL     string        `json:"metrics_url"`
	PprofURL       string        `json:"pprof_url"`
	RouterPID      int           `json:"router_pid"`
	OutDir         string        `json:"out_dir"`
	Model          string        `json:"model"`
	Quick          bool          `json:"quick"`
	Concurrency    int           `json:"concurrency"`
	Rounds         int           `json:"rounds"`
	RoundDuration  time.Duration `json:"round_duration"`
	QuietGap       time.Duration `json:"quiet_gap"`
	SampleInterval time.Duration `json:"sample_interval"`
	HighCardIDs    int           `json:"high_card_ids"`
	Streaming      bool          `json:"streaming"`
}

// Plan is Config with every derived duration resolved, so the summary records
// exactly what ran rather than the pre-quick-mode flag values.
type Plan struct {
	Config
	CalibrationSteps    []int         `json:"calibration_steps"`
	CalibrationDuration time.Duration `json:"calibration_duration"`
	WarmupDuration      time.Duration `json:"warmup_duration"`
	BaselineWindow      time.Duration `json:"baseline_window"`
}

// NewPlan resolves derived settings, collapsing every duration in quick mode so
// a full pass takes minutes instead of hours.
func NewPlan(cfg Config) Plan {
	p := Plan{
		Config:              cfg,
		CalibrationSteps:    []int{16, 64, 100},
		CalibrationDuration: 2 * time.Minute,
		WarmupDuration:      3 * time.Minute,
		BaselineWindow:      60 * time.Second,
	}
	if cfg.Quick {
		p.CalibrationSteps = []int{8, 32}
		p.CalibrationDuration = 30 * time.Second
		p.WarmupDuration = 45 * time.Second
		p.BaselineWindow = 20 * time.Second
		p.RoundDuration = 90 * time.Second
		p.QuietGap = 30 * time.Second
		p.HighCardIDs = 5000
	}
	return p
}

// CalibrationStep records one point of the concurrency knee sweep.
type CalibrationStep struct {
	Concurrency  int     `json:"concurrency"`
	ThroughputQP float64 `json:"throughput_qps"`
	P99LatencyMs float64 `json:"p99_latency_ms"`
	Successful   int     `json:"successful"`
	Failed       int     `json:"failed"`
}

// RoundStats is one measured phase.
type RoundStats struct {
	Name             string    `json:"name"`
	Phase            string    `json:"phase"`
	Concurrency      int       `json:"concurrency"`
	StartedAt        time.Time `json:"started_at"`
	FinishedAt       time.Time `json:"finished_at"`
	TotalRequests    int       `json:"total_requests"`
	Successful       int       `json:"successful"`
	Failed           int       `json:"failed"`
	ThroughputQPS    float64   `json:"throughput_qps"`
	P99LatencyMs     float64   `json:"p99_latency_ms"`
	PeakRSSBytes     float64   `json:"peak_rss_bytes"`
	SteadyRSSBytes   float64   `json:"steady_rss_bytes"`
	GCLiveBytes      float64   `json:"gc_live_bytes"`
	GoroutinesSteady float64   `json:"goroutines_steady"`
	Samples          int       `json:"samples"`
}

// RequestSummary records how much traffic the fixed-concurrency rounds served,
// so a reader can tell a plateau from a broken stack.
type RequestSummary struct {
	Successful  int     `json:"successful"`
	Total       int     `json:"total"`
	SuccessRate float64 `json:"success_rate"`
}

// Summary is the machine-readable record of a run.
type Summary struct {
	StartedAt    time.Time         `json:"started_at"`
	FinishedAt   time.Time         `json:"finished_at"`
	Interrupted  bool              `json:"interrupted"`
	Plan         Plan              `json:"plan"`
	Calibration  []CalibrationStep `json:"calibration"`
	Saturated    bool              `json:"saturated"`
	Warmup       RoundStats        `json:"warmup"`
	Rounds       []RoundStats      `json:"rounds"`
	Measured     RequestSummary    `json:"measured_requests"`
	HighCard     RoundStats        `json:"high_cardinality_round"`
	HighCardIDs  uint64            `json:"high_cardinality_unique_ids"`
	CapCrossed   bool              `json:"session_store_cap_crossed"`
	CapCrossedAt *time.Time        `json:"session_store_cap_crossed_at,omitempty"`
	ScrapeErrors int               `json:"scrape_errors"`
	Notes        []string          `json:"notes,omitempty"`
}

// Runner executes the full soak sequence against a running stack.
type Runner struct {
	plan    Plan
	sampler *Sampler
	client  *Client
	summary *Summary
}

// NewRunner wires up the sampler and HTTP client for a plan.
func NewRunner(plan Plan) (*Runner, error) {
	sampler, err := NewSampler(SamplerConfig{
		MetricsURL: plan.MetricsURL,
		PprofURL:   plan.PprofURL,
		OutDir:     plan.OutDir,
		Interval:   plan.SampleInterval,
		RouterPID:  plan.RouterPID,
	})
	if err != nil {
		return nil, err
	}
	return &Runner{
		plan:    plan,
		sampler: sampler,
		client:  NewClient(plan.GatewayURL, plan.Model, plan.peakConcurrency(), plan.HighCardIDs, plan.Streaming),
		summary: &Summary{StartedAt: time.Now().UTC(), Plan: plan},
	}, nil
}

func (p Plan) peakConcurrency() int {
	peak := p.Concurrency
	for _, c := range p.CalibrationSteps {
		if c > peak {
			peak = c
		}
	}
	return peak
}

// Run executes calibration, warmup, the fixed-concurrency rounds and the
// high-cardinality round.
func (r *Runner) Run(ctx context.Context) (err error) {
	sampleCtx, stopSampling := context.WithCancel(context.Background())
	r.sampler.Start(sampleCtx)

	defer func() {
		stopSampling()
		_ = r.sampler.Close()
		r.summary.FinishedAt = time.Now().UTC()
		r.summary.ScrapeErrors = r.sampler.ScrapeErrors()
		if writeErr := r.finalize(); writeErr != nil && err == nil {
			err = writeErr
		}
	}()

	if err := r.calibrate(ctx); err != nil {
		return r.interrupted(err)
	}
	if err := r.warmup(ctx); err != nil {
		return r.interrupted(err)
	}
	for i := 1; i <= r.plan.Rounds; i++ {
		stats, err := r.soakRound(ctx, fmt.Sprintf("round-%d", i), i, r.client.Chat)
		if err != nil {
			return r.interrupted(err)
		}
		r.summary.Rounds = append(r.summary.Rounds, stats)
		if err := r.quiet(ctx, i); err != nil {
			return r.interrupted(err)
		}
	}
	stats, err := r.soakRound(ctx, "highcard", r.plan.Rounds+1, r.client.ChatHighCardinality)
	if err != nil {
		return r.interrupted(err)
	}
	r.summary.HighCard = stats
	r.summary.HighCardIDs = r.client.UniqueIDsIssued()
	r.summary.CapCrossed = r.client.CapCrossed()
	if at := r.client.CapCrossedAt(); !at.IsZero() {
		r.summary.CapCrossedAt = &at
	}
	if !r.summary.CapCrossed {
		r.summary.Notes = append(r.summary.Notes,
			fmt.Sprintf("high-cardinality round issued %d unique IDs, below the %d per-store cap: eviction behaviour was not exercised",
				r.summary.HighCardIDs, sessionStoreCapEntries))
	}
	return nil
}

func (r *Runner) interrupted(err error) error {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		r.summary.Interrupted = true
		r.summary.Notes = append(r.summary.Notes, "run interrupted before completion; recorded series cover partial data only")
		return nil
	}
	return err
}

func (r *Runner) calibrate(ctx context.Context) error {
	for _, c := range r.plan.CalibrationSteps {
		if err := ctx.Err(); err != nil {
			return err
		}
		r.sampler.SetPhase(fmt.Sprintf("calibrate-%d", c), 0)
		log("calibration: concurrency=%d for %v", c, r.plan.CalibrationDuration)
		lg := performance.NewLoadGenerator(c, 0, r.plan.CalibrationDuration)
		res, err := lg.GenerateLoad(ctx, r.client.Chat)
		if err != nil {
			return err
		}
		r.summary.Calibration = append(r.summary.Calibration, CalibrationStep{
			Concurrency:  c,
			ThroughputQP: res.ThroughputQPS,
			P99LatencyMs: res.P99LatencyMs,
			Successful:   res.SuccessfulReqs,
			Failed:       res.FailedReqs,
		})
	}
	r.summary.Saturated = detectSaturation(r.summary.Calibration)
	return ctx.Err()
}

func detectSaturation(steps []CalibrationStep) bool {
	if len(steps) < 2 {
		return false
	}
	last, prev := steps[len(steps)-1], steps[len(steps)-2]
	if prev.ThroughputQP <= 0 || prev.P99LatencyMs <= 0 {
		return false
	}
	return last.ThroughputQP <= prev.ThroughputQP*1.05 && last.P99LatencyMs > prev.P99LatencyMs*3
}

func (r *Runner) warmup(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	r.sampler.SetPhase("warmup", 0)
	log("warmup: concurrency=%d for %v", r.plan.Concurrency, r.plan.WarmupDuration)
	lg := performance.NewLoadGenerator(r.plan.Concurrency, 0, r.plan.WarmupDuration)
	start := time.Now().UTC()
	res, err := lg.GenerateLoad(ctx, r.client.Chat)
	if err != nil {
		return err
	}
	r.summary.Warmup = r.statsFor("warmup", "warmup", r.plan.Concurrency, start, time.Now().UTC(), res)
	return ctx.Err()
}

func (r *Runner) soakRound(ctx context.Context, phase string, round int, reqFunc performance.RequestFunc) (RoundStats, error) {
	if err := ctx.Err(); err != nil {
		return RoundStats{}, err
	}
	r.sampler.SetPhase(phase, round)
	_ = r.sampler.Dump(ctx, phase+"-start")
	log("%s: concurrency=%d for %v", phase, r.plan.Concurrency, r.plan.RoundDuration)

	lg := performance.NewLoadGenerator(r.plan.Concurrency, 0, r.plan.RoundDuration)
	start := time.Now().UTC()
	res, err := lg.GenerateLoad(ctx, reqFunc)
	if err != nil {
		return RoundStats{}, err
	}
	end := time.Now().UTC()
	_ = r.sampler.Dump(ctx, phase+"-end")
	return r.statsFor(phase, phase, r.plan.Concurrency, start, end, res), ctx.Err()
}

func (r *Runner) quiet(ctx context.Context, round int) error {
	if r.plan.QuietGap <= 0 {
		return ctx.Err()
	}
	r.sampler.SetPhase(fmt.Sprintf("quiet-%d", round), round)
	log("quiet gap: %v", r.plan.QuietGap)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(r.plan.QuietGap):
		return nil
	}
}

func (r *Runner) statsFor(name, phase string, concurrency int, start, end time.Time, res *performance.LoadResult) RoundStats {
	stats := RoundStats{
		Name:          name,
		Phase:         phase,
		Concurrency:   concurrency,
		StartedAt:     start,
		FinishedAt:    end,
		TotalRequests: res.TotalRequests,
		Successful:    res.SuccessfulReqs,
		Failed:        res.FailedReqs,
		ThroughputQPS: res.ThroughputQPS,
		P99LatencyMs:  res.P99LatencyMs,
	}
	samples := r.phaseSamples(phase)
	stats.Samples = len(samples)
	if len(samples) == 0 {
		return stats
	}
	stats.PeakRSSBytes = MaxField(samples, func(s Sample) float64 { return s.RSS })
	stats.SteadyRSSBytes = TailMean(samples, r.plan.BaselineWindow, func(s Sample) float64 { return s.RSS })
	stats.GCLiveBytes = MinField(samples, func(s Sample) float64 { return s.HeapAlloc })
	stats.GoroutinesSteady = TailMean(samples, r.plan.BaselineWindow, func(s Sample) float64 { return s.Goroutines })
	return stats
}

func (r *Runner) phaseSamples(phase string) []Sample {
	return FilterSamples(r.sampler.Samples(), func(s Sample) bool { return s.Phase == phase })
}

func (r *Runner) finalize() error {
	var m RequestSummary
	for _, rd := range r.summary.Rounds {
		m.Successful += rd.Successful
		m.Total += rd.TotalRequests
	}
	if m.Total > 0 {
		m.SuccessRate = float64(m.Successful) / float64(m.Total)
	}
	r.summary.Measured = m
	if m.Total == 0 || m.SuccessRate < lowSuccessRateFloor {
		note := fmt.Sprintf(
			"only %d of %d fixed-round requests succeeded (%.0f%%, floor %.0f%%): the router served little or no traffic, so the recorded series describe a broken stack, not a baseline",
			m.Successful, m.Total, m.SuccessRate*100, lowSuccessRateFloor*100)
		r.summary.Notes = append(r.summary.Notes, note)
		fmt.Fprintf(os.Stderr, "soak: warning: %s\n", note)
	}
	if err := writeJSON(filepath.Join(r.plan.OutDir, "summary.json"), r.summary); err != nil {
		return err
	}
	rows := make([]BenchRow, 0, len(r.summary.Rounds)+1)
	for i, rd := range r.summary.Rounds {
		rows = append(rows, benchRowFor(fmt.Sprintf("Soak/round=%d", i+1), rd))
	}
	if r.summary.HighCard.Samples > 0 {
		rows = append(rows, benchRowFor("Soak/round=highcard", r.summary.HighCard))
	}
	config := map[string]string{
		"goos":          runtime.GOOS,
		"goarch":        runtime.GOARCH,
		"concurrency":   fmt.Sprintf("%d", r.plan.Concurrency),
		"mode":          modeName(r.plan.Quick),
		"response-mode": responseModeName(r.plan.Streaming),
	}
	return os.WriteFile(filepath.Join(r.plan.OutDir, "summary.bench"), []byte(FormatBench(config, rows)), 0o644)
}

func benchRowFor(name string, rd RoundStats) BenchRow {
	return BenchRow{
		Name:             name,
		Iterations:       1,
		PeakRSSBytes:     rd.PeakRSSBytes,
		SteadyRSSBytes:   rd.SteadyRSSBytes,
		GCLiveBytes:      rd.GCLiveBytes,
		GoroutinesSteady: rd.GoroutinesSteady,
		P99LatencyNs:     rd.P99LatencyMs * 1e6,
	}
}

func modeName(quick bool) string {
	if quick {
		return "quick"
	}
	return "full"
}

func responseModeName(streaming bool) string {
	if streaming {
		return "streaming"
	}
	return "buffered"
}

func writeJSON(path string, v any) error {
	body, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(body, '\n'), 0o644)
}

func log(format string, args ...any) {
	fmt.Printf("[soak %s] "+format+"\n", append([]any{time.Now().Format("15:04:05")}, args...)...)
}
