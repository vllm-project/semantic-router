package benchmark

import (
	"fmt"
	"html"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type TrendReport struct {
	Name        string        `json:"name"`
	Title       string        `json:"title"`
	Description string        `json:"description,omitempty"`
	Suite       string        `json:"suite"`
	XDimension  string        `json:"x_dimension"`
	Metric      string        `json:"metric"`
	Unit        string        `json:"unit"`
	XScale      string        `json:"x_scale"`
	File        string        `json:"file"`
	Series      []TrendSeries `json:"series"`
	SVG         string        `json:"-"`
}

type TrendSeries struct {
	Name   string       `json:"name"`
	Points []TrendPoint `json:"points"`
}

type TrendPoint struct {
	X         float64 `json:"x"`
	XLabel    string  `json:"x_label"`
	Value     float64 `json:"value"`
	Benchmark string  `json:"benchmark"`
}

func buildTrendReports(definitions []TrendMetadata, benchmarks map[string]BenchmarkMetric) []TrendReport {
	var reports []TrendReport
	for _, definition := range definitions {
		report, ok := buildTrendReport(definition, benchmarks)
		if ok {
			reports = append(reports, report)
		}
	}
	return reports
}

func buildTrendReport(definition TrendMetadata, benchmarks map[string]BenchmarkMetric) (TrendReport, bool) {
	pattern, err := regexp.Compile(definition.Benchmark)
	if err != nil {
		return TrendReport{}, false
	}
	seriesPoints, unit := collectTrendPoints(definition, pattern, benchmarks)
	report := TrendReport{
		Name: definition.Name, Title: definition.Title, Description: definition.Description,
		Suite: definition.Suite, XDimension: definition.XDimension,
		Metric: definition.Metric, Unit: unit, XScale: definition.XScale,
		File: definition.Name + ".svg",
	}
	if report.XScale == "" {
		report.XScale = "linear"
	}
	report.Series = sortedTrendSeries(seriesPoints)
	if len(report.Series) == 0 {
		return TrendReport{}, false
	}
	report.SVG = renderTrendSVG(report)
	return report, true
}

func collectTrendPoints(
	definition TrendMetadata,
	pattern *regexp.Regexp,
	benchmarks map[string]BenchmarkMetric,
) (map[string][]TrendPoint, string) {
	seriesPoints := make(map[string][]TrendPoint)
	unit := ""
	for name, metric := range benchmarks {
		seriesName, point, metricUnit, ok := trendPointForMetric(definition, pattern, name, metric)
		if !ok {
			continue
		}
		unit = metricUnit
		seriesPoints[seriesName] = append(seriesPoints[seriesName], point)
	}
	return seriesPoints, unit
}

func trendPointForMetric(
	definition TrendMetadata,
	pattern *regexp.Regexp,
	name string,
	metric BenchmarkMetric,
) (string, TrendPoint, string, bool) {
	if metric.Suite != definition.Suite || !pattern.MatchString(name) {
		return "", TrendPoint{}, "", false
	}
	xLabel, ok := metric.Dimensions[definition.XDimension]
	if !ok {
		return "", TrendPoint{}, "", false
	}
	x, err := strconv.ParseFloat(xLabel, 64)
	if err != nil {
		return "", TrendPoint{}, "", false
	}
	value, unit, ok := trendMetricValue(definition.Metric, metric)
	if !ok {
		return "", TrendPoint{}, "", false
	}
	seriesName, ok := trendSeriesName(definition.SeriesDimension, metric.Dimensions)
	return seriesName, TrendPoint{X: x, XLabel: xLabel, Value: value, Benchmark: name}, unit, ok
}

func trendSeriesName(dimension string, dimensions map[string]string) (string, bool) {
	if dimension == "" {
		return "current", true
	}
	value, ok := dimensions[dimension]
	return dimension + "=" + value, ok
}

func sortedTrendSeries(seriesPoints map[string][]TrendPoint) []TrendSeries {
	var names []string
	for name, points := range seriesPoints {
		if len(points) >= 2 {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	series := make([]TrendSeries, 0, len(names))
	for _, name := range names {
		points := seriesPoints[name]
		sort.Slice(points, func(i, j int) bool { return points[i].X < points[j].X })
		series = append(series, TrendSeries{Name: name, Points: points})
	}
	return series
}

func trendMetricValue(metricName string, metric BenchmarkMetric) (float64, string, bool) {
	switch metricName {
	case "latency_us_per_op":
		return metric.NsPerOp / 1_000, "µs/op", metric.NsPerOp > 0
	case "latency_ms_per_op":
		return metric.NsPerOp / 1_000_000, "ms/op", metric.NsPerOp > 0
	case "allocs_per_op":
		return float64(metric.AllocsPerOp), "allocs/op", true
	case "bytes_per_op":
		return float64(metric.BytesPerOp), "B/op", true
	case "p50_latency_ms":
		return metric.P50LatencyMs, "p50 ms", metric.P50LatencyMs > 0
	case "p95_latency_ms":
		return metric.P95LatencyMs, "p95 ms", metric.P95LatencyMs > 0
	case "p99_latency_ms":
		return metric.P99LatencyMs, "p99 ms", metric.P99LatencyMs > 0
	case "throughput_qps":
		return metric.ThroughputQPS, "requests/s", metric.ThroughputQPS > 0
	default:
		customName := strings.TrimPrefix(metricName, "custom:")
		value, ok := metric.Custom[customName]
		return value, customName, ok
	}
}

func (r *Report) SaveTrends(outputDir string) error {
	if err := saveJSON(filepath.Join(outputDir, "trends.json"), r.Trends); err != nil {
		return err
	}
	if len(r.Trends) == 0 {
		return nil
	}
	chartDir := filepath.Join(outputDir, "charts")
	if err := os.MkdirAll(chartDir, 0o755); err != nil {
		return fmt.Errorf("create trend chart directory: %w", err)
	}
	for _, trend := range r.Trends {
		if err := os.WriteFile(filepath.Join(chartDir, trend.File), []byte(trend.SVG), 0o644); err != nil {
			return fmt.Errorf("write trend chart %q: %w", trend.Name, err)
		}
	}
	return nil
}

type trendChartLayout struct {
	width, height, left, top, plotWidth, plotHeight float64
}

type trendChartBounds struct {
	minX, maxX, maxY float64
	xLabels          map[float64]string
}

func renderTrendSVG(trend TrendReport) string {
	layout := trendChartLayout{width: 960, height: 430, left: 78, top: 24, plotWidth: 692, plotHeight: 342}
	bounds := calculateTrendBounds(trend)
	xPos := func(value float64) float64 {
		return layout.left + (trendX(value, trend.XScale)-bounds.minX)/(bounds.maxX-bounds.minX)*layout.plotWidth
	}
	yPos := func(value float64) float64 {
		return layout.top + layout.plotHeight - value/bounds.maxY*layout.plotHeight
	}
	var svg strings.Builder
	svg.WriteString(fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %.0f %.0f" role="img" aria-label="%s">`, layout.width, layout.height, html.EscapeString(trend.Title)))
	svg.WriteString(`<rect width="100%" height="100%" fill="white"/><g font-family="system-ui,sans-serif" font-size="12" fill="#334155">`)
	writeTrendAxes(&svg, trend, layout, bounds, xPos, yPos)
	writeTrendSeries(&svg, trend, layout, xPos, yPos)
	svg.WriteString(`</g></svg>`)
	return svg.String()
}

func calculateTrendBounds(trend TrendReport) trendChartBounds {
	bounds := trendChartBounds{minX: math.MaxFloat64, maxX: -math.MaxFloat64, xLabels: make(map[float64]string)}
	for _, series := range trend.Series {
		for _, point := range series.Points {
			x := trendX(point.X, trend.XScale)
			bounds.minX = math.Min(bounds.minX, x)
			bounds.maxX = math.Max(bounds.maxX, x)
			bounds.maxY = math.Max(bounds.maxY, point.Value)
			bounds.xLabels[point.X] = point.XLabel
		}
	}
	if bounds.minX == bounds.maxX {
		bounds.maxX++
	}
	if bounds.maxY <= 0 {
		bounds.maxY = 1
	}
	bounds.maxY *= 1.1
	return bounds
}

func writeTrendAxes(
	svg *strings.Builder,
	trend TrendReport,
	layout trendChartLayout,
	bounds trendChartBounds,
	xPos, yPos func(float64) float64,
) {
	for tick := 0; tick <= 5; tick++ {
		value := bounds.maxY * float64(tick) / 5
		y := yPos(value)
		svg.WriteString(fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#e2e8f0"/><text x="%.1f" y="%.1f" text-anchor="end">%s</text>`, layout.left, y, layout.left+layout.plotWidth, y, layout.left-9, y+4, formatTrendNumber(value)))
	}
	xValues := sortedTrendXValues(bounds.xLabels)
	for _, value := range xValues {
		x := xPos(value)
		svg.WriteString(fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#e2e8f0"/><text x="%.1f" y="%.1f" text-anchor="middle">%s</text>`, x, layout.top, x, layout.top+layout.plotHeight, x, layout.top+layout.plotHeight+21, html.EscapeString(bounds.xLabels[value])))
	}
	svg.WriteString(fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#64748b"/><line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#64748b"/>`, layout.left, layout.top+layout.plotHeight, layout.left+layout.plotWidth, layout.top+layout.plotHeight, layout.left, layout.top, layout.left, layout.top+layout.plotHeight))
	svg.WriteString(fmt.Sprintf(`<text x="%.1f" y="%.1f" text-anchor="middle" font-weight="600">%s%s</text>`, layout.left+layout.plotWidth/2, layout.height-12, html.EscapeString(trend.XDimension), trendScaleSuffix(trend.XScale)))
	svg.WriteString(fmt.Sprintf(`<text transform="translate(18 %.1f) rotate(-90)" text-anchor="middle" font-weight="600">%s</text>`, layout.top+layout.plotHeight/2, html.EscapeString(trend.Unit)))
}

func sortedTrendXValues(labels map[float64]string) []float64 {
	values := make([]float64, 0, len(labels))
	for value := range labels {
		values = append(values, value)
	}
	sort.Float64s(values)
	return values
}

func writeTrendSeries(
	svg *strings.Builder,
	trend TrendReport,
	layout trendChartLayout,
	xPos, yPos func(float64) float64,
) {
	colors := []string{"#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4b5563"}
	for index, series := range trend.Series {
		color := colors[index%len(colors)]
		svg.WriteString(trendSeriesPath(series, color, xPos, yPos))
		for _, point := range series.Points {
			svg.WriteString(fmt.Sprintf(`<circle cx="%.1f" cy="%.1f" r="4" fill="%s"><title>%s: %s %s at %s=%s</title></circle>`, xPos(point.X), yPos(point.Value), color, html.EscapeString(series.Name), formatTrendNumber(point.Value), html.EscapeString(trend.Unit), html.EscapeString(trend.XDimension), html.EscapeString(point.XLabel)))
		}
		legendY := layout.top + float64(index)*24
		svg.WriteString(fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="3"/><text x="%.1f" y="%.1f">%s</text>`, layout.left+layout.plotWidth+22, legendY+6, layout.left+layout.plotWidth+42, legendY+6, color, layout.left+layout.plotWidth+49, legendY+10, html.EscapeString(series.Name)))
	}
}

func trendSeriesPath(
	series TrendSeries,
	color string,
	xPos, yPos func(float64) float64,
) string {
	var path strings.Builder
	for index, point := range series.Points {
		command := "L"
		if index == 0 {
			command = "M"
		}
		path.WriteString(fmt.Sprintf("%s%.1f %.1f ", command, xPos(point.X), yPos(point.Value)))
	}
	return fmt.Sprintf(`<path d="%s" fill="none" stroke="%s" stroke-width="2.5"/>`, path.String(), color)
}

func trendX(value float64, scale string) float64 {
	if scale == "log2" && value > 0 {
		return math.Log2(value)
	}
	return value
}

func trendScaleSuffix(scale string) string {
	if scale == "log2" {
		return " (log2 scale)"
	}
	return ""
}

func formatTrendNumber(value float64) string {
	absolute := math.Abs(value)
	switch {
	case absolute >= 1_000_000:
		return fmt.Sprintf("%.1fM", value/1_000_000)
	case absolute >= 1_000:
		return fmt.Sprintf("%.1fk", value/1_000)
	case absolute >= 10:
		return fmt.Sprintf("%.1f", value)
	default:
		return fmt.Sprintf("%.2f", value)
	}
}
