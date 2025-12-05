package benchmark

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Report represents a performance report
type Report struct {
	Metadata       ReportMetadata     `json:"metadata"`
	Comparisons    []ComparisonResult `json:"comparisons"`
	HasRegressions bool               `json:"has_regressions"`
	Summary        ReportSummary      `json:"summary"`
}

// ReportMetadata holds metadata about the report
type ReportMetadata struct {
	GeneratedAt time.Time `json:"generated_at"`
	GitCommit   string    `json:"git_commit"`
	GitBranch   string    `json:"git_branch"`
	GoVersion   string    `json:"go_version"`
}

// ReportSummary holds summary statistics
type ReportSummary struct {
	TotalBenchmarks   int `json:"total_benchmarks"`
	RegressionsFound  int `json:"regressions_found"`
	ImprovementsFound int `json:"improvements_found"`
	NoChangeFound     int `json:"no_change_found"`
}

// GenerateReport creates a performance report from comparison results
func GenerateReport(comparisons []ComparisonResult, metadata ReportMetadata) *Report {
	report := &Report{
		Metadata:       metadata,
		Comparisons:    comparisons,
		HasRegressions: HasRegressions(comparisons),
	}

	// Calculate summary
	for _, comp := range comparisons {
		report.Summary.TotalBenchmarks++
		if comp.RegressionDetected {
			report.Summary.RegressionsFound++
		} else if comp.NsPerOpChange < -5 { // 5% improvement threshold
			report.Summary.ImprovementsFound++
		} else {
			report.Summary.NoChangeFound++
		}
	}

	return report
}

// SaveJSON saves the report as JSON
func (r *Report) SaveJSON(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("failed to create report directory: %w", err)
	}

	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal report: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write report file: %w", err)
	}

	fmt.Printf("JSON report saved: %s\n", path)
	return nil
}

// SaveMarkdown saves the report as Markdown
func (r *Report) SaveMarkdown(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("failed to create report directory: %w", err)
	}

	var md strings.Builder

	// Header
	md.WriteString("# Performance Benchmark Report\n\n")
	md.WriteString(fmt.Sprintf("**Generated:** %s\n\n", r.Metadata.GeneratedAt.Format(time.RFC3339)))
	md.WriteString(fmt.Sprintf("**Git Commit:** %s\n\n", r.Metadata.GitCommit))
	md.WriteString(fmt.Sprintf("**Git Branch:** %s\n\n", r.Metadata.GitBranch))
	md.WriteString(fmt.Sprintf("**Go Version:** %s\n\n", r.Metadata.GoVersion))

	// Summary
	md.WriteString("## Summary\n\n")
	md.WriteString(fmt.Sprintf("- **Total Benchmarks:** %d\n", r.Summary.TotalBenchmarks))
	md.WriteString(fmt.Sprintf("- **Regressions:** %d\n", r.Summary.RegressionsFound))
	md.WriteString(fmt.Sprintf("- **Improvements:** %d\n", r.Summary.ImprovementsFound))
	md.WriteString(fmt.Sprintf("- **No Change:** %d\n\n", r.Summary.NoChangeFound))

	if r.HasRegressions {
		md.WriteString("⚠️ **WARNING: Performance regressions detected!**\n\n")
	} else {
		md.WriteString("✅ **No regressions detected**\n\n")
	}

	// Detailed results
	md.WriteString("## Detailed Results\n\n")
	md.WriteString("| Benchmark | Metric | Baseline | Current | Change | Status |\n")
	md.WriteString("|-----------|--------|----------|---------|--------|--------|\n")

	for _, comp := range r.Comparisons {
		status := "✅ OK"
		if comp.RegressionDetected {
			status = "⚠️ REGRESSION"
		} else if comp.NsPerOpChange < -5 {
			status = "🚀 IMPROVED"
		}

		// ns/op row
		md.WriteString(fmt.Sprintf("| %s | ns/op | %d | %d | %+.2f%% | %s |\n",
			comp.BenchmarkName,
			comp.Baseline.NsPerOp,
			comp.Current.NsPerOp,
			comp.NsPerOpChange,
			status,
		))

		// P95 latency row if available
		if comp.Baseline.P95LatencyMs > 0 {
			md.WriteString(fmt.Sprintf("| %s | P95 Latency | %.2fms | %.2fms | %+.2f%% | |\n",
				"",
				comp.Baseline.P95LatencyMs,
				comp.Current.P95LatencyMs,
				comp.P95LatencyChange,
			))
		}

		// Throughput row if available
		if comp.Baseline.ThroughputQPS > 0 {
			md.WriteString(fmt.Sprintf("| %s | Throughput | %.2f qps | %.2f qps | %+.2f%% | |\n",
				"",
				comp.Baseline.ThroughputQPS,
				comp.Current.ThroughputQPS,
				comp.ThroughputChange,
			))
		}
	}

	if err := os.WriteFile(path, []byte(md.String()), 0644); err != nil {
		return fmt.Errorf("failed to write markdown report: %w", err)
	}

	fmt.Printf("Markdown report saved: %s\n", path)
	return nil
}

// SaveHTML saves the report as HTML
func (r *Report) SaveHTML(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("failed to create report directory: %w", err)
	}

	var html strings.Builder

	html.WriteString(`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; }
        h1 { color: #333; }
        .metadata { background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
        .summary-card { background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }
        .summary-card.regression { background-color: #ffe8e8; }
        .summary-card.improvement { background-color: #e8ffe8; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .regression { color: #d32f2f; font-weight: bold; }
        .improvement { color: #388e3c; font-weight: bold; }
        .ok { color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Performance Benchmark Report</h1>
`)

	// Metadata
	html.WriteString(`        <div class="metadata">`)
	html.WriteString(fmt.Sprintf(`            <p><strong>Generated:</strong> %s</p>`, r.Metadata.GeneratedAt.Format(time.RFC3339)))
	html.WriteString(fmt.Sprintf(`            <p><strong>Git Commit:</strong> %s</p>`, r.Metadata.GitCommit))
	html.WriteString(fmt.Sprintf(`            <p><strong>Git Branch:</strong> %s</p>`, r.Metadata.GitBranch))
	html.WriteString(fmt.Sprintf(`            <p><strong>Go Version:</strong> %s</p>`, r.Metadata.GoVersion))
	html.WriteString(`        </div>`)

	// Summary
	html.WriteString(`        <div class="summary">`)
	html.WriteString(fmt.Sprintf(`            <div class="summary-card"><h3>%d</h3><p>Total Benchmarks</p></div>`, r.Summary.TotalBenchmarks))
	html.WriteString(fmt.Sprintf(`            <div class="summary-card regression"><h3>%d</h3><p>Regressions</p></div>`, r.Summary.RegressionsFound))
	html.WriteString(fmt.Sprintf(`            <div class="summary-card improvement"><h3>%d</h3><p>Improvements</p></div>`, r.Summary.ImprovementsFound))
	html.WriteString(fmt.Sprintf(`            <div class="summary-card"><h3>%d</h3><p>No Change</p></div>`, r.Summary.NoChangeFound))
	html.WriteString(`        </div>`)

	// Results table
	html.WriteString(`        <table>`)
	html.WriteString(`            <tr><th>Benchmark</th><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th><th>Status</th></tr>`)

	for _, comp := range r.Comparisons {
		statusClass := "ok"
		statusText := "OK"
		if comp.RegressionDetected {
			statusClass = "regression"
			statusText = "REGRESSION"
		} else if comp.NsPerOpChange < -5 {
			statusClass = "improvement"
			statusText = "IMPROVED"
		}

		html.WriteString(fmt.Sprintf(`            <tr><td>%s</td><td>ns/op</td><td>%d</td><td>%d</td><td>%+.2f%%</td><td class="%s">%s</td></tr>`,
			comp.BenchmarkName,
			comp.Baseline.NsPerOp,
			comp.Current.NsPerOp,
			comp.NsPerOpChange,
			statusClass,
			statusText,
		))
	}

	html.WriteString(`        </table>`)
	html.WriteString(`    </div>`)
	html.WriteString(`</body>`)
	html.WriteString(`</html>`)

	if err := os.WriteFile(path, []byte(html.String()), 0644); err != nil {
		return fmt.Errorf("failed to write HTML report: %w", err)
	}

	fmt.Printf("HTML report saved: %s\n", path)
	return nil
}
