package benchmark

import (
	"fmt"
	"html/template"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type Report struct {
	Metadata         RunMetadata   `json:"metadata"`
	BaselineMetadata RunMetadata   `json:"baseline_metadata"`
	Summary          ReportSummary `json:"summary"`
	Rows             []ReportRow   `json:"rows"`
	Trends           []TrendReport `json:"trends,omitempty"`
	Ungated          []string      `json:"ungated_benchmarks"`
	Missing          []string      `json:"missing_benchmarks"`
	HasRegressions   bool          `json:"has_regressions"`
	CoverageComplete bool          `json:"coverage_complete"`
}

type ReportSummary struct {
	Measured         int `json:"measured"`
	Compared         int `json:"compared"`
	Regressions      int `json:"regressions"`
	TimingAdvisories int `json:"timing_advisories"`
	Unbaselined      int `json:"unbaselined"`
	Missing          int `json:"missing"`
}

type ReportRow struct {
	Suite                      string             `json:"suite"`
	Benchmark                  string             `json:"benchmark"`
	Dimensions                 map[string]string  `json:"dimensions,omitempty"`
	Samples                    int                `json:"samples"`
	BaselineNsPerOp            float64            `json:"baseline_ns_per_op"`
	CurrentNsPerOp             float64            `json:"current_ns_per_op"`
	NsChange                   float64            `json:"ns_change_percent"`
	NsStdDev                   float64            `json:"ns_stddev_percent"`
	BaselineAllocs             int64              `json:"baseline_allocs_per_op"`
	CurrentAllocs              int64              `json:"current_allocs_per_op"`
	AllocsChange               float64            `json:"allocs_change_percent"`
	BaselineBytes              int64              `json:"baseline_bytes_per_op"`
	CurrentBytes               int64              `json:"current_bytes_per_op"`
	BytesChange                float64            `json:"bytes_change_percent"`
	BaselineCustom             map[string]float64 `json:"baseline_custom,omitempty"`
	CurrentCustom              map[string]float64 `json:"current_custom,omitempty"`
	CurrentP50Ms               float64            `json:"current_p50_latency_ms,omitempty"`
	BaselineP95Ms              float64            `json:"baseline_p95_latency_ms,omitempty"`
	CurrentP95Ms               float64            `json:"current_p95_latency_ms,omitempty"`
	CurrentP99Ms               float64            `json:"current_p99_latency_ms,omitempty"`
	P95Change                  float64            `json:"p95_latency_change_percent,omitempty"`
	BaselineQPS                float64            `json:"baseline_throughput_qps,omitempty"`
	CurrentQPS                 float64            `json:"current_throughput_qps,omitempty"`
	ThroughputChange           float64            `json:"throughput_change_percent,omitempty"`
	BaselineUpstreamCalls      float64            `json:"baseline_upstream_calls,omitempty"`
	CurrentUpstreamCalls       float64            `json:"current_upstream_calls,omitempty"`
	UpstreamCallsChange        float64            `json:"upstream_calls_change_percent,omitempty"`
	BaselineTokenAmplification float64            `json:"baseline_token_amplification,omitempty"`
	CurrentTokenAmplification  float64            `json:"current_token_amplification,omitempty"`
	TokenAmplificationChange   float64            `json:"token_amplification_change_percent,omitempty"`
	Status                     string             `json:"status"`
}

func GenerateReport(comparison *ComparisonDocument) *Report {
	report := &Report{
		Metadata:         comparison.CurrentMetadata,
		BaselineMetadata: comparison.BaselineMetadata,
		Ungated:          comparison.Ungated,
		Missing:          comparison.Missing,
		HasRegressions:   comparison.HasRegressions,
		CoverageComplete: comparison.CoverageComplete,
	}
	if len(comparison.CurrentBenchmarks) > 0 {
		report.Summary.Measured = len(comparison.CurrentBenchmarks)
	} else {
		for _, suite := range comparison.CurrentMetadata.Suites {
			report.Summary.Measured += suite.BenchmarkCount
		}
	}
	report.Summary.Compared = len(comparison.Results)
	report.Summary.Unbaselined = len(comparison.Ungated)
	report.Summary.Missing = len(comparison.Missing)
	compared := make(map[string]struct{}, len(comparison.Results))
	for _, result := range comparison.Results {
		compared[result.BenchmarkName] = struct{}{}
		status := "ok"
		if result.RegressionDetected {
			status = "regression"
			report.Summary.Regressions++
		} else if result.NsAdvisory {
			status = "timing-advisory"
			report.Summary.TimingAdvisories++
		}
		report.Rows = append(report.Rows, ReportRow{
			Suite:                      result.Suite,
			Benchmark:                  result.BenchmarkName,
			Dimensions:                 result.Current.Dimensions,
			Samples:                    result.Current.Samples,
			BaselineNsPerOp:            result.Baseline.NsPerOp,
			CurrentNsPerOp:             result.Current.NsPerOp,
			NsChange:                   result.NsPerOpChange,
			NsStdDev:                   result.Current.NsStdDevPct,
			BaselineAllocs:             result.Baseline.AllocsPerOp,
			CurrentAllocs:              result.Current.AllocsPerOp,
			AllocsChange:               result.AllocsPerOpChange,
			BaselineBytes:              result.Baseline.BytesPerOp,
			CurrentBytes:               result.Current.BytesPerOp,
			BytesChange:                result.BytesPerOpChange,
			BaselineCustom:             result.Baseline.Custom,
			CurrentCustom:              result.Current.Custom,
			CurrentP50Ms:               result.Current.P50LatencyMs,
			BaselineP95Ms:              result.Baseline.P95LatencyMs,
			CurrentP95Ms:               result.Current.P95LatencyMs,
			CurrentP99Ms:               result.Current.P99LatencyMs,
			P95Change:                  result.P95LatencyChange,
			BaselineQPS:                result.Baseline.ThroughputQPS,
			CurrentQPS:                 result.Current.ThroughputQPS,
			ThroughputChange:           result.ThroughputChange,
			BaselineUpstreamCalls:      result.Baseline.UpstreamCalls,
			CurrentUpstreamCalls:       result.Current.UpstreamCalls,
			UpstreamCallsChange:        result.UpstreamCallsChange,
			BaselineTokenAmplification: result.Baseline.TokenAmplification,
			CurrentTokenAmplification:  result.Current.TokenAmplification,
			TokenAmplificationChange:   result.TokenAmplificationChange,
			Status:                     status,
		})
	}
	for name, current := range comparison.CurrentBenchmarks {
		if _, ok := compared[name]; ok {
			continue
		}
		report.Rows = append(report.Rows, ReportRow{
			Suite: current.Suite, Benchmark: name, Dimensions: current.Dimensions,
			Samples: current.Samples, CurrentNsPerOp: current.NsPerOp,
			NsStdDev: current.NsStdDevPct, CurrentAllocs: current.AllocsPerOp,
			CurrentBytes: current.BytesPerOp, CurrentCustom: current.Custom,
			CurrentP50Ms: current.P50LatencyMs, CurrentP95Ms: current.P95LatencyMs,
			CurrentP99Ms: current.P99LatencyMs, CurrentQPS: current.ThroughputQPS,
			CurrentUpstreamCalls:      current.UpstreamCalls,
			CurrentTokenAmplification: current.TokenAmplification,
			Status:                    "unbaselined",
		})
	}
	sort.Slice(report.Rows, func(i, j int) bool {
		if report.Rows[i].Suite != report.Rows[j].Suite {
			return report.Rows[i].Suite < report.Rows[j].Suite
		}
		return report.Rows[i].Benchmark < report.Rows[j].Benchmark
	})
	report.Trends = buildTrendReports(comparison.CurrentMetadata.Trends, comparison.CurrentBenchmarks)
	return report
}

func (r *Report) SaveAll(outputDir string) error {
	if err := r.SaveJSON(filepath.Join(outputDir, "report.json")); err != nil {
		return err
	}
	if err := r.SaveMarkdown(filepath.Join(outputDir, "report.md")); err != nil {
		return err
	}
	if err := r.SaveTrends(outputDir); err != nil {
		return err
	}
	return r.SaveHTML(filepath.Join(outputDir, "report.html"))
}

func (r *Report) SaveJSON(path string) error {
	return saveJSON(path, r)
}

func (r *Report) SaveMarkdown(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create report directory: %w", err)
	}
	var output strings.Builder
	output.WriteString("# Performance report\n\n")
	output.WriteString(fmt.Sprintf("Environment: `%s` (%s)· Profile: `%s`· Commit: `%s`\n\n",
		r.Metadata.Environment, r.Metadata.EnvironmentKind, r.Metadata.Profile, r.Metadata.GitCommit))
	output.WriteString(fmt.Sprintf("Host: `%s/%s`· Go: `%s`· CPU: `%s` (%d cores)\n\n",
		r.Metadata.GOOS, r.Metadata.GOARCH, r.Metadata.GoVersion, r.Metadata.CPUModel, r.Metadata.CPUCount))

	result := "PASS"
	if r.HasRegressions || !r.CoverageComplete {
		result = "FAIL"
	}
	output.WriteString("## Summary\n\n")
	output.WriteString(fmt.Sprintf("Result: **%s**· measured %d· compared %d· regressions %d· timing advisories %d\n\n",
		result, r.Summary.Measured, r.Summary.Compared, r.Summary.Regressions, r.Summary.TimingAdvisories))
	output.WriteString("Allocation count and bytes/op are portable component gates. External suites gate p95 latency, throughput, upstream calls, and token amplification. ns/op is advisory unless a same-runner timing gate is enabled.\n\n")

	if len(r.Ungated) > 0 {
		output.WriteString("### Unbaselined measurements\n\n")
		for _, name := range r.Ungated {
			output.WriteString("- `" + strings.ReplaceAll(name, "`", "") + "`\n")
		}
		output.WriteString("\n")
	}
	if len(r.Missing) > 0 {
		output.WriteString("### Missing measurements\n\n")
		for _, name := range r.Missing {
			output.WriteString("- `" + strings.ReplaceAll(name, "`", "") + "`\n")
		}
		output.WriteString("\n")
	}

	writeMarkdownTrends(&output, r.Trends)

	output.WriteString("## Measurements\n\n")
	output.WriteString("| Suite | Benchmark | Samples | ns/op base → current | Δ time | CV | allocs/op base → current | B/op base → current | custom metrics base → current | p95 ms base → current | QPS base → current | upstream calls | token amplification | Status |\n")
	output.WriteString("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |\n")
	for _, row := range r.Rows {
		output.WriteString(fmt.Sprintf("| %s | %s | %d | %.1f → %.1f | %+.1f%% | %.1f%% | %d → %d | %d → %d | %s | %s | %s | %s | %s | %s |\n",
			markdownCell(row.Suite), markdownCell(row.Benchmark), row.Samples,
			row.BaselineNsPerOp, row.CurrentNsPerOp, row.NsChange, row.NsStdDev,
			row.BaselineAllocs, row.CurrentAllocs, row.BaselineBytes, row.CurrentBytes,
			markdownCell(optionalCustomMetricPairs(row.BaselineCustom, row.CurrentCustom)),
			optionalMetricPair(row.BaselineP95Ms, row.CurrentP95Ms),
			optionalMetricPair(row.BaselineQPS, row.CurrentQPS),
			optionalMetricPair(row.BaselineUpstreamCalls, row.CurrentUpstreamCalls),
			optionalMetricPair(row.BaselineTokenAmplification, row.CurrentTokenAmplification), row.Status))
	}

	if err := os.WriteFile(path, []byte(output.String()), 0o644); err != nil {
		return fmt.Errorf("write Markdown report: %w", err)
	}
	return nil
}

func writeMarkdownTrends(output *strings.Builder, trends []TrendReport) {
	if len(trends) == 0 {
		return
	}
	output.WriteString("## Scaling trends\n\n")
	output.WriteString("Trend charts compare current-run points on the same host. The endpoint table remains visible in the GitHub job summary; download the artifact for the SVGs and inline HTML charts. Read each description before comparing model-backed and model-isolated suites.\n\n")
	for _, trend := range trends {
		writeMarkdownTrend(output, trend)
	}
}

func writeMarkdownTrend(output *strings.Builder, trend TrendReport) {
	output.WriteString("### " + trend.Title + "\n\n")
	if trend.Description != "" {
		output.WriteString(trend.Description + "\n\n")
	}
	output.WriteString(fmt.Sprintf("![%s](charts/%s)\n\n", markdownCell(trend.Title), trend.File))
	output.WriteString("| Series | First point | Last point | Endpoint Δ |\n")
	output.WriteString("| --- | ---: | ---: | ---: |\n")
	for _, series := range trend.Series {
		first := series.Points[0]
		last := series.Points[len(series.Points)-1]
		change := trendEndpointChange(first.Value, last.Value)
		output.WriteString(fmt.Sprintf("| %s | %s=%s: %s %s | %s=%s: %s %s | %+.1f%% |\n",
			markdownCell(series.Name), trend.XDimension, first.XLabel,
			formatTrendNumber(first.Value), trend.Unit, trend.XDimension, last.XLabel,
			formatTrendNumber(last.Value), trend.Unit, change))
	}
	output.WriteString("\n")
}

func trendEndpointChange(first, last float64) float64 {
	if first == 0 {
		return 0
	}
	return (last - first) / first * 100
}

func markdownCell(value string) string {
	value = strings.ReplaceAll(value, "|", "\\|")
	return strings.ReplaceAll(value, "\n", " ")
}

func optionalMetricPair(baseline, current float64) string {
	if baseline == 0 && current == 0 {
		return "—"
	}
	return fmt.Sprintf("%.1f → %.1f", baseline, current)
}

func optionalCustomMetricPairs(baseline, current map[string]float64) string {
	keys := make(map[string]struct{}, len(baseline)+len(current))
	for key := range baseline {
		keys[key] = struct{}{}
	}
	for key := range current {
		keys[key] = struct{}{}
	}
	if len(keys) == 0 {
		return "—"
	}

	sortedKeys := make([]string, 0, len(keys))
	for key := range keys {
		sortedKeys = append(sortedKeys, key)
	}
	sort.Strings(sortedKeys)

	pairs := make([]string, 0, len(sortedKeys))
	for _, key := range sortedKeys {
		pairs = append(pairs, fmt.Sprintf("%s=%s → %s", key,
			customMetricValue(baseline, key), customMetricValue(current, key)))
	}
	return strings.Join(pairs, "; ")
}

func customMetricValue(metrics map[string]float64, key string) string {
	value, ok := metrics[key]
	if !ok {
		return "—"
	}
	return fmt.Sprintf("%.1f", value)
}

func (r *Report) SaveHTML(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create report directory: %w", err)
	}
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create HTML report: %w", err)
	}
	defer file.Close()
	tmpl, err := template.New("report").Funcs(template.FuncMap{
		"change": func(value float64) string { return fmt.Sprintf("%+.1f%%", value) },
		"number": func(value float64) string { return fmt.Sprintf("%.1f", value) },
		"pair":   optionalMetricPair,
		"custom": optionalCustomMetricPairs,
		"svg":    func(value string) template.HTML { return template.HTML(value) }, // #nosec G203 -- generated locally from escaped benchmark data.
	}).Parse(reportHTMLTemplate)
	if err != nil {
		return fmt.Errorf("parse HTML report template: %w", err)
	}
	if err := tmpl.Execute(file, r); err != nil {
		return fmt.Errorf("render HTML report: %w", err)
	}
	return nil
}

const reportHTMLTemplate = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic Router performance report</title>
<style>
body{font:14px system-ui,sans-serif;margin:2rem;color:#172033}h1{margin-bottom:.3rem}.meta{color:#536079;margin-bottom:1.5rem}
.cards{display:flex;gap:1rem;flex-wrap:wrap}.card{background:#f3f6fa;border-radius:8px;padding:.8rem 1.2rem;min-width:110px}.bad{color:#b42318;font-weight:700}.warn{color:#9a6700}.ok{color:#067647;font-weight:700}
table{border-collapse:collapse;width:100%;margin-top:1.5rem}th,td{border-bottom:1px solid #d8dee9;padding:.55rem;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}code{font-size:12px}
.trend-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:1rem}.trend{border:1px solid #d8dee9;border-radius:8px;padding:1rem}.trend h3{margin:.1rem 0}.trend p{color:#536079}.trend svg{width:100%;height:auto}
</style></head><body>
<h1>Semantic Router performance report</h1>
<div class="meta">{{.Metadata.Environment}} / {{.Metadata.Profile}} · {{.Metadata.GitCommit}} · {{.Metadata.GOOS}}/{{.Metadata.GOARCH}} · {{.Metadata.GoVersion}}</div>
<div class="cards"><div class="card"><strong>{{.Summary.Measured}}</strong><br>measured</div><div class="card"><strong>{{.Summary.Compared}}</strong><br>compared</div><div class="card"><strong>{{.Summary.Regressions}}</strong><br>regressions</div><div class="card"><strong>{{.Summary.TimingAdvisories}}</strong><br>timing advisories</div></div>
{{if .Ungated}}<h2 class="bad">Unbaselined measurements</h2><ul>{{range .Ungated}}<li><code>{{.}}</code></li>{{end}}</ul>{{end}}
{{if .Missing}}<h2 class="bad">Missing measurements</h2><ul>{{range .Missing}}<li><code>{{.}}</code></li>{{end}}</ul>{{end}}
{{if .Trends}}<h2>Scaling trends</h2><p>Current-run curves from one host; model-isolated and model-backed suites are labeled separately.</p><div class="trend-grid">{{range .Trends}}<section class="trend"><h3>{{.Title}}</h3><p>{{.Description}}</p>{{svg .SVG}}</section>{{end}}</div>{{end}}
<table><thead><tr><th>Suite</th><th>Benchmark</th><th>samples</th><th>base ns/op</th><th>current ns/op</th><th>Δ time</th><th>CV</th><th>allocs/op</th><th>B/op</th><th>custom metrics</th><th>p95 ms</th><th>QPS</th><th>upstream calls</th><th>token amp</th><th>status</th></tr></thead><tbody>
{{range .Rows}}<tr><td>{{.Suite}}</td><td><code>{{.Benchmark}}</code></td><td>{{.Samples}}</td><td>{{number .BaselineNsPerOp}}</td><td>{{number .CurrentNsPerOp}}</td><td>{{change .NsChange}}</td><td>{{number .NsStdDev}}%</td><td>{{.BaselineAllocs}} → {{.CurrentAllocs}}</td><td>{{.BaselineBytes}} → {{.CurrentBytes}}</td><td>{{custom .BaselineCustom .CurrentCustom}}</td><td>{{pair .BaselineP95Ms .CurrentP95Ms}}</td><td>{{pair .BaselineQPS .CurrentQPS}}</td><td>{{pair .BaselineUpstreamCalls .CurrentUpstreamCalls}}</td><td>{{pair .BaselineTokenAmplification .CurrentTokenAmplification}}</td><td class="{{if eq .Status "regression"}}bad{{else if eq .Status "timing-advisory"}}warn{{else}}ok{{end}}">{{.Status}}</td></tr>{{end}}
</tbody></table></body></html>`
