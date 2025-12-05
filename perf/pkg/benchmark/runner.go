package benchmark

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"time"
)

// Runner orchestrates benchmark execution and profiling
type Runner struct {
	config    *Config
	profiler  *Profiler
	collector *MetricsCollector
}

// NewRunner creates a new benchmark runner
func NewRunner(configPath string) (*Runner, error) {
	config, err := LoadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	profiler := NewProfiler(config.Profiling.OutputDir)
	collector := NewMetricsCollector()

	return &Runner{
		config:    config,
		profiler:  profiler,
		collector: collector,
	}, nil
}

// RunBenchmarks executes all benchmarks with profiling
func (r *Runner) RunBenchmarks(ctx context.Context, suites []string) (*BenchmarkResults, error) {
	fmt.Printf("Starting benchmark run at %s\n", time.Now().Format(time.RFC3339))
	fmt.Printf("Go version: %s\n", runtime.Version())
	fmt.Printf("GOOS: %s, GOARCH: %s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("CPU cores: %d\n\n", runtime.NumCPU())

	results := &BenchmarkResults{
		StartTime: time.Now(),
		Suites:    make(map[string]*SuiteResult),
	}

	// Start profiling if enabled
	if r.config.Profiling.EnableCPU {
		if err := r.profiler.StartCPU(); err != nil {
			return nil, fmt.Errorf("failed to start CPU profiling: %w", err)
		}
		defer r.profiler.StopCPU()
	}

	// Collect baseline metrics
	baselineMetrics := r.collector.Collect()
	results.BaselineMetrics = baselineMetrics

	// Run benchmark suites
	for _, suite := range suites {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			fmt.Printf("Running benchmark suite: %s\n", suite)
			// Suite execution will be handled by Go's testing framework
			// This runner orchestrates the overall process
		}
	}

	// Take memory snapshot if enabled
	if r.config.Profiling.EnableMemory {
		if err := r.profiler.TakeMemSnapshot(); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to take memory snapshot: %v\n", err)
		}
	}

	// Take goroutine snapshot if enabled
	if r.config.Profiling.EnableGoroutine {
		if err := r.profiler.TakeGoroutineSnapshot(); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to take goroutine snapshot: %v\n", err)
		}
	}

	// Collect final metrics
	finalMetrics := r.collector.Collect()
	results.FinalMetrics = finalMetrics

	results.EndTime = time.Now()
	results.Duration = results.EndTime.Sub(results.StartTime)

	return results, nil
}

// BenchmarkResults holds all benchmark execution results
type BenchmarkResults struct {
	StartTime       time.Time
	EndTime         time.Time
	Duration        time.Duration
	Suites          map[string]*SuiteResult
	BaselineMetrics *RuntimeMetrics
	FinalMetrics    *RuntimeMetrics
}

// SuiteResult holds results for a single benchmark suite
type SuiteResult struct {
	Name      string
	Duration  time.Duration
	TestCount int
	Passed    int
	Failed    int
}

// Profiler handles pprof profiling
type Profiler struct {
	outputDir string
	cpuFile   *os.File
}

// NewProfiler creates a new profiler
func NewProfiler(outputDir string) *Profiler {
	return &Profiler{
		outputDir: outputDir,
	}
}

// MetricsCollector collects runtime metrics
type MetricsCollector struct{}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{}
}

// RuntimeMetrics holds runtime performance metrics
type RuntimeMetrics struct {
	Timestamp      time.Time
	CPUCount       int
	GoroutineCount int
	MemStats       runtime.MemStats
}

// Collect gathers current runtime metrics
func (mc *MetricsCollector) Collect() *RuntimeMetrics {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	return &RuntimeMetrics{
		Timestamp:      time.Now(),
		CPUCount:       runtime.NumCPU(),
		GoroutineCount: runtime.NumGoroutine(),
		MemStats:       memStats,
	}
}
