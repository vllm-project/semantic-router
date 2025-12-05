package profiler

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"time"
)

// Profiler manages pprof profiling operations
type Profiler struct {
	outputDir string
	cpuFile   *os.File
}

// New creates a new profiler instance
func New(outputDir string) *Profiler {
	return &Profiler{
		outputDir: outputDir,
	}
}

// StartCPU begins CPU profiling
func (p *Profiler) StartCPU() error {
	if err := os.MkdirAll(p.outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	filename := filepath.Join(p.outputDir, fmt.Sprintf("cpu-%s.prof", time.Now().Format("20060102-150405")))
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create CPU profile file: %w", err)
	}

	if err := pprof.StartCPUProfile(f); err != nil {
		f.Close()
		return fmt.Errorf("failed to start CPU profiling: %w", err)
	}

	p.cpuFile = f
	fmt.Printf("CPU profiling started: %s\n", filename)
	return nil
}

// StopCPU stops CPU profiling
func (p *Profiler) StopCPU() error {
	if p.cpuFile == nil {
		return nil
	}

	pprof.StopCPUProfile()
	if err := p.cpuFile.Close(); err != nil {
		return fmt.Errorf("failed to close CPU profile file: %w", err)
	}

	fmt.Printf("CPU profiling stopped: %s\n", p.cpuFile.Name())
	p.cpuFile = nil
	return nil
}

// TakeMemSnapshot takes a memory profile snapshot
func (p *Profiler) TakeMemSnapshot() error {
	if err := os.MkdirAll(p.outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	filename := filepath.Join(p.outputDir, fmt.Sprintf("mem-%s.prof", time.Now().Format("20060102-150405")))
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create memory profile file: %w", err)
	}
	defer f.Close()

	runtime.GC() // Get up-to-date statistics
	if err := pprof.WriteHeapProfile(f); err != nil {
		return fmt.Errorf("failed to write heap profile: %w", err)
	}

	fmt.Printf("Memory snapshot saved: %s\n", filename)
	return nil
}

// TakeGoroutineSnapshot takes a goroutine profile snapshot
func (p *Profiler) TakeGoroutineSnapshot() error {
	if err := os.MkdirAll(p.outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	filename := filepath.Join(p.outputDir, fmt.Sprintf("goroutine-%s.prof", time.Now().Format("20060102-150405")))
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create goroutine profile file: %w", err)
	}
	defer f.Close()

	if err := pprof.Lookup("goroutine").WriteTo(f, 0); err != nil {
		return fmt.Errorf("failed to write goroutine profile: %w", err)
	}

	fmt.Printf("Goroutine snapshot saved: %s\n", filename)
	return nil
}

// TakeBlockSnapshot takes a block profile snapshot
func (p *Profiler) TakeBlockSnapshot() error {
	runtime.SetBlockProfileRate(1) // Enable block profiling

	if err := os.MkdirAll(p.outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	filename := filepath.Join(p.outputDir, fmt.Sprintf("block-%s.prof", time.Now().Format("20060102-150405")))
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create block profile file: %w", err)
	}
	defer f.Close()

	if err := pprof.Lookup("block").WriteTo(f, 0); err != nil {
		return fmt.Errorf("failed to write block profile: %w", err)
	}

	fmt.Printf("Block snapshot saved: %s\n", filename)
	return nil
}

// TakeMutexSnapshot takes a mutex profile snapshot
func (p *Profiler) TakeMutexSnapshot() error {
	runtime.SetMutexProfileFraction(1) // Enable mutex profiling

	if err := os.MkdirAll(p.outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	filename := filepath.Join(p.outputDir, fmt.Sprintf("mutex-%s.prof", time.Now().Format("20060102-150405")))
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create mutex profile file: %w", err)
	}
	defer f.Close()

	if err := pprof.Lookup("mutex").WriteTo(f, 0); err != nil {
		return fmt.Errorf("failed to write mutex profile: %w", err)
	}

	fmt.Printf("Mutex snapshot saved: %s\n", filename)
	return nil
}
