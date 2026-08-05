package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/perf/pkg/benchmark"
)

func main() {
	// Command-line flags
	compareBaseline := flag.String("compare-baseline", "", "Path to baseline directory")
	currentResults := flag.String("current", "", "Path to current benchmark results (JSON)")
	thresholdFile := flag.String("threshold-file", "", "Path to thresholds configuration file")
	outputPath := flag.String("output", "", "Output path for reports")
	generateReport := flag.Bool("generate-report", false, "Generate performance report")
	inputPath := flag.String("input", "", "Input comparison JSON for report generation")
	parseBench := flag.String("parse-bench", "", "Path to raw `go test -bench` output to convert into a current-results JSON (writes to --output)")
	failOnRegression := flag.Bool("fail-on-regression", false, "Exit non-zero if any benchmark regresses beyond its configured threshold")

	flag.Parse()

	if *parseBench != "" {
		if err := parseBenchToBaseline(*parseBench, *outputPath); err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing benchmark output: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if *generateReport {
		if *inputPath == "" {
			fmt.Fprintln(os.Stderr, "Error: --input required for report generation")
			os.Exit(1)
		}
		if err := generateReportFromComparison(*inputPath, *outputPath); err != nil {
			fmt.Fprintf(os.Stderr, "Error generating report: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if *compareBaseline != "" {
		if err := compareWithBaseline(*compareBaseline, *currentResults, *thresholdFile, *outputPath, *failOnRegression); err != nil {
			fmt.Fprintf(os.Stderr, "Error comparing with baseline: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// Default: print help
	fmt.Println("Performance Testing Tool")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  perftest --compare-baseline=<dir> --current=<file> --threshold-file=<file> --output=<file>")
	fmt.Println("  perftest --generate-report --input=<file> --output=<file>")
	fmt.Println()
	flag.PrintDefaults()
}

func compareWithBaseline(baselineDir, currentResultsFile, thresholdFile, outputPath string, failOnRegression bool) error {
	fmt.Println("Comparing performance with baseline...")
	fmt.Printf("Baseline directory: %s\n", baselineDir)
	fmt.Printf("Current results: %s\n", currentResultsFile)
	fmt.Printf("Threshold file: %s\n", thresholdFile)

	// Load thresholds
	var thresholds *benchmark.ThresholdsConfig
	var err error
	if thresholdFile != "" {
		thresholds, err = benchmark.LoadThresholds(thresholdFile)
		if err != nil {
			return fmt.Errorf("failed to load thresholds: %w", err)
		}
	}

	// Load baseline: union every per-suite *.json in the directory.
	// update-baseline.sh writes one file per suite (decision.json, looper.json,
	// …), never a combined baseline.json, so we merge them here (#2455 rc#1).
	baseline, err := benchmark.LoadBaselineDir(baselineDir)
	if err != nil {
		return fmt.Errorf("failed to load baseline: %w", err)
	}
	fmt.Printf("Loaded baseline with %d benchmarks\n", len(baseline.Benchmarks))

	current, err := loadCurrentResults(currentResultsFile, baselineDir)
	if err != nil {
		return err
	}
	fmt.Printf("Loaded current results with %d benchmarks\n", len(current.Benchmarks))

	// Surface benchmarks with no baseline so a pass is not mistaken for full
	// coverage (e.g. classification/cache before their baselines are populated).
	if ungated := benchmark.UngatedBenchmarks(current, baseline); len(ungated) > 0 {
		fmt.Printf("ℹ️  %d benchmark(s) had no baseline and were NOT gated: %s\n",
			len(ungated), strings.Join(ungated, ", "))
	}

	// Surface baseline benchmarks that produced no current result: they were
	// expected but did not run (a crashed suite, a rename, or a filtered run),
	// so they are not compared and a disappeared benchmark would pass silently.
	if missing := benchmark.MissingBenchmarks(current, baseline); len(missing) > 0 {
		fmt.Printf("⚠️  %d baseline benchmark(s) had NO current result (not measured — crashed, renamed, or filtered?): %s\n",
			len(missing), strings.Join(missing, ", "))
	}

	results, err := benchmark.CompareWithBaseline(current, baseline, thresholds)
	if err != nil {
		return fmt.Errorf("failed to compare results: %w", err)
	}
	benchmark.PrintComparisonResults(results)

	if err := writeComparisonOutput(outputPath, baselineDir, currentResultsFile, results); err != nil {
		return err
	}

	if benchmark.HasRegressions(results) {
		n := countRegressions(results)
		fmt.Printf("\n⚠️  %d performance regression(s) detected beyond configured thresholds!\n", n)
		if failOnRegression {
			return fmt.Errorf("%d performance regression(s) exceeded the configured thresholds", n)
		}
	}

	return nil
}

// loadCurrentResults reads the current benchmark results from an explicit file,
// or from the current.json fallback in the baseline directory.
func loadCurrentResults(currentResultsFile, baselineDir string) (*benchmark.Baseline, error) {
	path := currentResultsFile
	if path == "" {
		path = filepath.Join(baselineDir, "current.json")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read current results: %w", err)
	}
	var current benchmark.Baseline
	if err := json.Unmarshal(data, &current); err != nil {
		return nil, fmt.Errorf("failed to parse current results: %w", err)
	}
	return &current, nil
}

// writeComparisonOutput marshals the comparison to outputPath as JSON when a
// path is given; a empty path is a no-op.
func writeComparisonOutput(outputPath, baselineDir, currentResultsFile string, results []benchmark.ComparisonResult) error {
	if outputPath == "" {
		return nil
	}
	comparisonOutput := struct {
		BaselineDir    string                       `json:"baseline_dir"`
		CurrentFile    string                       `json:"current_file"`
		Timestamp      time.Time                    `json:"timestamp"`
		Results        []benchmark.ComparisonResult `json:"results"`
		HasRegressions bool                         `json:"has_regressions"`
	}{
		BaselineDir:    baselineDir,
		CurrentFile:    currentResultsFile,
		Timestamp:      time.Now(),
		Results:        results,
		HasRegressions: benchmark.HasRegressions(results),
	}
	outputData, err := json.MarshalIndent(comparisonOutput, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal comparison: %w", err)
	}
	if err := os.WriteFile(outputPath, outputData, 0o644); err != nil {
		return fmt.Errorf("failed to write output: %w", err)
	}
	fmt.Printf("Comparison results saved to: %s\n", outputPath)
	return nil
}

func countRegressions(results []benchmark.ComparisonResult) int {
	n := 0
	for _, r := range results {
		if r.RegressionDetected {
			n++
		}
	}
	return n
}

// parseBenchToBaseline reads raw `go test -bench` output and writes it as a
// Baseline JSON (the "current" result set). This is what makes a comparable
// current/baseline pair possible: benchmarks run now are captured in the same
// schema as the recorded baselines (#2455 rc#2).
func parseBenchToBaseline(inputPath, outputPath string) error {
	if outputPath == "" {
		return fmt.Errorf("--output is required with --parse-bench")
	}

	f, err := os.Open(inputPath)
	if err != nil {
		return fmt.Errorf("failed to open benchmark output: %w", err)
	}
	defer f.Close()

	baseline, err := benchmark.ParseBenchOutput(f)
	if err != nil {
		return err
	}
	if len(baseline.Benchmarks) == 0 {
		return fmt.Errorf("no benchmark results found in %s", inputPath)
	}

	baseline.Version = "current"
	baseline.GitCommit = getGitCommit()
	baseline.Timestamp = time.Now()

	if err := benchmark.SaveBaseline(baseline, outputPath); err != nil {
		return err
	}
	fmt.Printf("Parsed %d benchmarks from %s -> %s\n", len(baseline.Benchmarks), inputPath, outputPath)
	return nil
}

func generateReportFromComparison(inputPath, outputPath string) error {
	fmt.Println("Generating performance report...")
	fmt.Printf("Input: %s\n", inputPath)
	fmt.Printf("Output: %s\n", outputPath)

	// Create report metadata
	metadata := benchmark.ReportMetadata{
		GeneratedAt: time.Now(),
		GitCommit:   getGitCommit(),
		GitBranch:   getGitBranch(),
		GoVersion:   runtime.Version(),
	}

	// Load comparison results from input file
	// For now, create empty report
	report := benchmark.GenerateReport([]benchmark.ComparisonResult{}, metadata)

	// Save in the format chosen from the output extension.
	if err := saveReport(report, outputPath); err != nil {
		return err
	}

	fmt.Println("Report generated successfully")
	return nil
}

// saveReport writes the report to outputPath, choosing the format from the file
// extension (defaulting to JSON). An empty path is a no-op.
func saveReport(report *benchmark.Report, outputPath string) error {
	switch {
	case outputPath == "":
		return nil
	case strings.HasSuffix(outputPath, ".md"):
		return report.SaveMarkdown(outputPath)
	case strings.HasSuffix(outputPath, ".html"):
		return report.SaveHTML(outputPath)
	case strings.HasSuffix(outputPath, ".json"):
		return report.SaveJSON(outputPath)
	default:
		return report.SaveJSON(outputPath + ".json")
	}
}

func getGitCommit() string {
	// This would use exec.Command to run: git rev-parse HEAD
	return "unknown"
}

func getGitBranch() string {
	// This would use exec.Command to run: git rev-parse --abbrev-ref HEAD
	return "unknown"
}
