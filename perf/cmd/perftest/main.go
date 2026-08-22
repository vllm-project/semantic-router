package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/perf/pkg/benchmark"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "perftest: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return usageError()
	}
	switch args[0] {
	case "validate":
		return validateCommand(args[1:])
	case "run":
		return runCommand(args[1:])
	case "compare":
		return compareCommand(args[1:])
	case "report":
		return reportCommand(args[1:])
	case "promote":
		return promoteCommand(args[1:])
	case "help", "-h", "--help":
		printUsage()
		return nil
	default:
		return fmt.Errorf("unknown command %q\n\n%s", args[0], usageText)
	}
}

func validateCommand(args []string) error {
	flags := flag.NewFlagSet("validate", flag.ContinueOnError)
	configPath := flags.String("config", "config/perf.yaml", "performance manifest")
	thresholdPath := flags.String("thresholds", "config/thresholds.yaml", "threshold policy")
	if err := flags.Parse(args); err != nil {
		return err
	}
	manifest, err := benchmark.LoadManifest(*configPath)
	if err != nil {
		return err
	}
	if _, err := benchmark.LoadThresholds(*thresholdPath); err != nil {
		return err
	}
	fmt.Printf("valid performance contract: schema=%d environments=%d profiles=%d suites=%d\n",
		manifest.SchemaVersion, len(manifest.Environments), len(manifest.Profiles), len(manifest.Suites))
	return nil
}

func runCommand(args []string) error {
	flags := flag.NewFlagSet("run", flag.ContinueOnError)
	configPath := flags.String("config", "config/perf.yaml", "performance manifest")
	thresholdPath := flags.String("thresholds", "config/thresholds.yaml", "threshold policy")
	repoRoot := flags.String("repo-root", "..", "repository root")
	environment := flags.String("environment", "cpu", "manifest environment")
	profile := flags.String("profile", "quick", "manifest profile")
	outputDir := flags.String("output-dir", "../reports/perf", "report directory")
	baselinePath := flags.String("baseline", "", "reviewed baseline JSON")
	failOnRegression := flags.Bool("fail-on-regression", false, "fail when a blocking threshold is exceeded")
	requireComplete := flags.Bool("require-complete", false, "fail for unbaselined or missing measurements")
	if err := flags.Parse(args); err != nil {
		return err
	}

	manifest, err := benchmark.LoadManifest(*configPath)
	if err != nil {
		return err
	}
	thresholds, err := benchmark.LoadThresholds(*thresholdPath)
	if err != nil {
		return err
	}
	runner, err := benchmark.NewRunner(manifest, *repoRoot)
	if err != nil {
		return err
	}
	current, err := runner.Run(context.Background(), benchmark.RunOptions{
		Environment: *environment,
		Profile:     *profile,
		OutputDir:   *outputDir,
	})
	if err != nil {
		return err
	}

	baseline := &benchmark.Baseline{Benchmarks: map[string]benchmark.BenchmarkMetric{}}
	if *baselinePath != "" {
		baseline, err = benchmark.LoadBaseline(*baselinePath)
		if err != nil {
			return err
		}
	}
	return compareAndReport(current, baseline, thresholds, *outputDir, *failOnRegression, *requireComplete)
}

func compareCommand(args []string) error {
	flags := flag.NewFlagSet("compare", flag.ContinueOnError)
	currentPath := flags.String("current", "", "current result JSON")
	baselinePath := flags.String("baseline", "", "reviewed baseline JSON")
	thresholdPath := flags.String("thresholds", "config/thresholds.yaml", "threshold policy")
	outputDir := flags.String("output-dir", "../reports/perf", "report directory")
	failOnRegression := flags.Bool("fail-on-regression", false, "fail when a blocking threshold is exceeded")
	requireComplete := flags.Bool("require-complete", false, "fail for unbaselined or missing measurements")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *currentPath == "" || *baselinePath == "" {
		return fmt.Errorf("compare requires --current and --baseline")
	}
	current, err := benchmark.LoadBaseline(*currentPath)
	if err != nil {
		return err
	}
	baseline, err := benchmark.LoadBaseline(*baselinePath)
	if err != nil {
		return err
	}
	thresholds, err := benchmark.LoadThresholds(*thresholdPath)
	if err != nil {
		return err
	}
	return compareAndReport(current, baseline, thresholds, *outputDir, *failOnRegression, *requireComplete)
}

func compareAndReport(
	current, baseline *benchmark.Baseline,
	thresholds *benchmark.ThresholdsConfig,
	outputDir string,
	failOnRegression, requireComplete bool,
) error {
	comparison, err := benchmark.BuildComparison(current, baseline, thresholds)
	if err != nil {
		return err
	}
	if err := benchmark.SaveComparison(comparison, filepath.Join(outputDir, "comparison.json")); err != nil {
		return err
	}
	report := benchmark.GenerateReport(comparison)
	if err := report.SaveAll(outputDir); err != nil {
		return err
	}
	benchmark.PrintComparisonResults(comparison)
	fmt.Printf("Reports: %s\n", outputDir)

	var failures []string
	if failOnRegression && comparison.HasRegressions {
		failures = append(failures, "blocking performance regression")
	}
	if requireComplete && !comparison.CoverageComplete {
		failures = append(failures, fmt.Sprintf("incomplete coverage (%d unbaselined, %d missing)", len(comparison.Ungated), len(comparison.Missing)))
	}
	if len(failures) > 0 {
		return errors.New(strings.Join(failures, "; "))
	}
	return nil
}

func reportCommand(args []string) error {
	flags := flag.NewFlagSet("report", flag.ContinueOnError)
	inputPath := flags.String("input", "", "comparison JSON")
	outputDir := flags.String("output-dir", "../reports/perf", "report directory")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *inputPath == "" {
		return fmt.Errorf("report requires --input")
	}
	data, err := os.ReadFile(*inputPath)
	if err != nil {
		return fmt.Errorf("read comparison: %w", err)
	}
	var comparison benchmark.ComparisonDocument
	if err := json.Unmarshal(data, &comparison); err != nil {
		return fmt.Errorf("parse comparison: %w", err)
	}
	return benchmark.GenerateReport(&comparison).SaveAll(*outputDir)
}

func promoteCommand(args []string) error {
	flags := flag.NewFlagSet("promote", flag.ContinueOnError)
	currentPath := flags.String("current", "", "reviewed current result JSON")
	outputPath := flags.String("output", "", "baseline destination")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *currentPath == "" || *outputPath == "" {
		return fmt.Errorf("promote requires --current and --output")
	}
	current, err := benchmark.LoadBaseline(*currentPath)
	if err != nil {
		return err
	}
	if len(current.Benchmarks) == 0 {
		return fmt.Errorf("refusing to promote an empty result set")
	}
	if err := benchmark.SaveBaseline(current, *outputPath); err != nil {
		return err
	}
	fmt.Printf("Promoted %d reviewed measurements to %s\n", len(current.Benchmarks), *outputPath)
	return nil
}

func usageError() error {
	return errors.New(usageText)
}

func printUsage() {
	fmt.Print(usageText)
}

const usageText = `Semantic Router performance harness

Usage:
  perftest validate [flags]
  perftest run [flags]
  perftest compare --current FILE --baseline FILE [flags]
  perftest report --input FILE [flags]
  perftest promote --current FILE --output FILE
`
