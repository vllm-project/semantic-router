package benchmark

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

type RunOptions struct {
	Environment string
	Profile     string
	OutputDir   string
}

type Command struct {
	Name string
	Args []string
	Dir  string
	Env  []string
}

type CommandExecutor interface {
	Run(context.Context, Command) ([]byte, error)
}

type OSCommandExecutor struct{}

func (OSCommandExecutor) Run(ctx context.Context, command Command) ([]byte, error) {
	cmd := exec.CommandContext(ctx, command.Name, command.Args...)
	cmd.Dir = command.Dir
	cmd.Env = command.Env
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	err := cmd.Run()
	return output.Bytes(), err
}

// Runner executes the resolved manifest suites. It has no CPU/GPU branching:
// the selected environment supplies capabilities and variables, while each
// suite owns its producer. That keeps future accelerator runners additive.
type Runner struct {
	manifest *Manifest
	repoRoot string
	executor CommandExecutor
}

func NewRunner(manifest *Manifest, repoRoot string) (*Runner, error) {
	return newRunnerWithExecutor(manifest, repoRoot, OSCommandExecutor{})
}

func newRunnerWithExecutor(manifest *Manifest, repoRoot string, executor CommandExecutor) (*Runner, error) {
	if manifest == nil {
		return nil, fmt.Errorf("performance manifest is nil")
	}
	absoluteRoot, err := filepath.Abs(repoRoot)
	if err != nil {
		return nil, fmt.Errorf("resolve repository root: %w", err)
	}
	if executor == nil {
		return nil, fmt.Errorf("command executor is nil")
	}
	return &Runner{manifest: manifest, repoRoot: absoluteRoot, executor: executor}, nil
}

func (r *Runner) Run(ctx context.Context, options RunOptions) (*Baseline, error) {
	resolved, err := r.manifest.Resolve(options.Environment, options.Profile)
	if err != nil {
		return nil, err
	}
	outputDir, err := filepath.Abs(options.OutputDir)
	if err != nil {
		return nil, fmt.Errorf("resolve output directory: %w", err)
	}
	if err := os.MkdirAll(filepath.Join(outputDir, "suites"), 0o755); err != nil {
		return nil, fmt.Errorf("create suite output directory: %w", err)
	}

	result := &Baseline{
		SchemaVersion: ResultSchemaVersion,
		Metadata:      collectRunMetadata(r.repoRoot, resolved),
		Benchmarks:    make(map[string]BenchmarkMetric),
	}
	fmt.Printf("Performance run: environment=%s profile=%s suites=%d\n", resolved.EnvironmentName, resolved.ProfileName, len(resolved.Suites))

	for _, suite := range resolved.Suites {
		started := time.Now()
		fmt.Printf("\n[%s] %s\n", suite.Name, suite.Config.Description)
		metrics, rawOutput, runErr := r.runSuite(ctx, resolved, suite, outputDir)
		rawPath := filepath.Join(outputDir, "suites", suite.Name+".log")
		if err := os.WriteFile(rawPath, rawOutput, 0o644); err != nil {
			return nil, fmt.Errorf("write raw output for suite %q: %w", suite.Name, err)
		}
		if len(rawOutput) > 0 {
			fmt.Print(string(rawOutput))
		}
		if runErr != nil {
			return nil, fmt.Errorf("suite %q failed (raw output: %s): %w", suite.Name, rawPath, runErr)
		}
		if len(metrics.Benchmarks) == 0 {
			return nil, fmt.Errorf("suite %q produced no benchmark measurements", suite.Name)
		}
		for name, metric := range metrics.Benchmarks {
			if _, duplicate := result.Benchmarks[name]; duplicate {
				return nil, fmt.Errorf("benchmark %q was emitted by more than one suite", name)
			}
			metric.Suite = suite.Name
			result.Benchmarks[name] = metric
		}
		result.Metadata.Suites = append(result.Metadata.Suites, SuiteRunMetadata{
			Name:            suite.Name,
			Runner:          suite.Config.Runner,
			DurationSeconds: time.Since(started).Seconds(),
			BenchmarkCount:  len(metrics.Benchmarks),
			Dimensions:      suite.Config.Dimensions,
		})
	}

	currentPath := filepath.Join(outputDir, "current.json")
	if err := SaveBaseline(result, currentPath); err != nil {
		return nil, err
	}
	fmt.Printf("\nCaptured %d benchmark measurements in %s\n", len(result.Benchmarks), currentPath)
	return result, nil
}

func (r *Runner) runSuite(
	ctx context.Context,
	resolved *ResolvedRun,
	suite ResolvedSuite,
	outputDir string,
) (*Baseline, []byte, error) {
	moduleDir, err := r.moduleDir(suite.Config.Module)
	if err != nil {
		return nil, nil, err
	}
	environment := append([]string(nil), os.Environ()...)
	for key, value := range resolved.Environment.Env {
		environment = append(environment, key+"="+value)
	}
	environment = append(environment,
		"VSR_PERF_ENVIRONMENT="+resolved.EnvironmentName,
		"VSR_PERF_PROFILE="+resolved.ProfileName,
		"VSR_PERF_SUITE="+suite.Name,
	)

	switch suite.Config.Runner {
	case "go_benchmark":
		args := []string{
			"test",
			"-run", "^$",
			"-bench", suite.Config.Benchmark,
			"-benchmem",
			"-count", strconv.Itoa(resolved.Profile.Count),
			"-benchtime", resolved.Profile.BenchTime,
			"-timeout", resolved.Profile.Timeout,
		}
		args = append(args, suite.Config.Packages...)
		output, commandErr := r.executor.Run(ctx, Command{Name: "go", Args: args, Dir: moduleDir, Env: environment})
		if commandErr != nil {
			return nil, output, commandErr
		}
		metrics, parseErr := ParseBenchOutputForSuite(bytes.NewReader(output), suite.Name)
		return metrics, output, parseErr
	case "external":
		resultPath := filepath.Join(outputDir, "suites", suite.Name+".json")
		environment = append(environment, "VSR_PERF_RESULT_FILE="+resultPath)
		output, commandErr := r.executor.Run(ctx, Command{
			Name: suite.Config.Command[0], Args: suite.Config.Command[1:], Dir: moduleDir, Env: environment,
		})
		if commandErr != nil {
			return nil, output, commandErr
		}
		metrics, loadErr := LoadBaseline(resultPath)
		if loadErr != nil {
			return nil, output, fmt.Errorf("external suite must write VSR_PERF_RESULT_FILE: %w", loadErr)
		}
		for name, metric := range metrics.Benchmarks {
			metric.Suite = suite.Name
			metrics.Benchmarks[name] = metric
		}
		return metrics, output, nil
	default:
		return nil, nil, fmt.Errorf("unsupported suite runner %q", suite.Config.Runner)
	}
}

func (r *Runner) moduleDir(module string) (string, error) {
	if filepath.IsAbs(module) {
		return "", fmt.Errorf("suite module must be repository-relative: %s", module)
	}
	dir := filepath.Clean(filepath.Join(r.repoRoot, module))
	relative, err := filepath.Rel(r.repoRoot, dir)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("suite module escapes repository root: %s", module)
	}
	if info, err := os.Stat(dir); err != nil || !info.IsDir() {
		return "", fmt.Errorf("suite module directory does not exist: %s", dir)
	}
	return dir, nil
}

func collectRunMetadata(repoRoot string, resolved *ResolvedRun) RunMetadata {
	return RunMetadata{
		GeneratedAt:     time.Now().UTC(),
		GitCommit:       commandOutput(repoRoot, "git", "rev-parse", "HEAD"),
		GitBranch:       commandOutput(repoRoot, "git", "rev-parse", "--abbrev-ref", "HEAD"),
		Environment:     resolved.EnvironmentName,
		EnvironmentKind: resolved.Environment.Kind,
		Accelerator:     resolved.Environment.Accelerator,
		Profile:         resolved.ProfileName,
		GoVersion:       runtime.Version(),
		GOOS:            runtime.GOOS,
		GOARCH:          runtime.GOARCH,
		CPUModel:        cpuModel(),
		CPUCount:        runtime.NumCPU(),
	}
}

func commandOutput(dir, name string, args ...string) string {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	output, err := cmd.Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(output))
}

func cpuModel() string {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		if key, value, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(key) == "model name" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
