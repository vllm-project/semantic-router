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

	flag.Parse()

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
		if err := compareWithBaseline(*compareBaseline, *currentResults, *thresholdFile, *outputPath); err != nil {
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

func compareWithBaseline(baselineDir, currentResultsFile, thresholdFile, outputPath string) error {
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

	// Load baseline
	baselinePath := filepath.Join(baselineDir, "baseline.json")
	baseline, err := benchmark.LoadBaseline(baselinePath)
	if err != nil {
		return fmt.Errorf("failed to load baseline: %w", err)
	}
	fmt.Printf("Loaded baseline with %d benchmarks\n", len(baseline.Benchmarks))

	// Load current results
	var current *benchmark.Baseline
	if currentResultsFile != "" {
		currentData, err := os.ReadFile(currentResultsFile)
		if err != nil {
			return fmt.Errorf("failed to read current results: %w", err)
		}
		if err := json.Unmarshal(currentData, &current); err != nil {
			return fmt.Errorf("failed to parse current results: %w", err)
		}
	} else {
		// Look for current.json in baseline dir
		currentPath := filepath.Join(baselineDir, "current.json")
		currentData, err := os.ReadFile(currentPath)
		if err != nil {
			return fmt.Errorf("failed to read current results: %w", err)
		}
		if err := json.Unmarshal(currentData, &current); err != nil {
			return fmt.Errorf("failed to parse current results: %w", err)
		}
	}
	fmt.Printf("Loaded current results with %d benchmarks\n", len(current.Benchmarks))

	// Compare with baseline
	results, err := benchmark.CompareWithBaseline(current, baseline, thresholds)
	if err != nil {
		return fmt.Errorf("failed to compare results: %w", err)
	}

	// Print results
	benchmark.PrintComparisonResults(results)

	// Generate comparison output
	comparisonOutput := struct {
		BaselineDir    string                    `json:"baseline_dir"`
		CurrentFile    string                    `json:"current_file"`
		Timestamp      time.Time                 `json:"timestamp"`
		Results        []benchmark.ComparisonResult `json:"results"`
		HasRegressions bool                      `json:"has_regressions"`
	}{
		BaselineDir:    baselineDir,
		CurrentFile:    currentResultsFile,
		Timestamp:      time.Now(),
		Results:        results,
		HasRegressions: benchmark.HasRegressions(results),
	}

	// Save comparison output
	if outputPath != "" {
		outputData, err := json.MarshalIndent(comparisonOutput, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to marshal comparison: %w", err)
		}
		if err := os.WriteFile(outputPath, outputData, 0644); err != nil {
			return fmt.Errorf("failed to write output: %w", err)
		}
		fmt.Printf("Comparison results saved to: %s\n", outputPath)
	}

	if benchmark.HasRegressions(results) {
		fmt.Println("\n⚠️ WARNING: Performance regressions detected!")
	}

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

	// Save in requested format based on output extension
	if outputPath != "" {
		if strings.HasSuffix(outputPath, ".json") {
			if err := report.SaveJSON(outputPath); err != nil {
				return err
			}
		} else if strings.HasSuffix(outputPath, ".md") {
			if err := report.SaveMarkdown(outputPath); err != nil {
				return err
			}
		} else if strings.HasSuffix(outputPath, ".html") {
			if err := report.SaveHTML(outputPath); err != nil {
				return err
			}
		} else {
			// Default to JSON
			if err := report.SaveJSON(outputPath + ".json"); err != nil {
				return err
			}
		}
	}

	fmt.Println("Report generated successfully")
	return nil
}

func getGitCommit() string {
	// This would use exec.Command to run: git rev-parse HEAD
	return "unknown"
}

func getGitBranch() string {
	// This would use exec.Command to run: git rev-parse --abbrev-ref HEAD
	return "unknown"
}
