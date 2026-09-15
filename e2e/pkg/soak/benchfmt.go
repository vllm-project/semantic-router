package soak

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// BenchRow is one round rendered as a Go benchmark result line.
type BenchRow struct {
	Name             string
	Iterations       int
	PeakRSSBytes     float64
	SteadyRSSBytes   float64
	GCLiveBytes      float64
	GoroutinesSteady float64
	P99LatencyNs     float64
}

var benchUnits = []string{
	"peak-RSS-bytes",
	"steady-RSS-bytes",
	"gc-live-bytes",
	"goroutines-steady",
	"p99-latency-ns",
}

// FormatBench renders rows in Go benchmark text format, directly consumable by
// benchstat. config holds optional `key: value` header lines.
func FormatBench(config map[string]string, rows []BenchRow) string {
	var b strings.Builder
	for _, k := range sortedKeys(config) {
		fmt.Fprintf(&b, "%s: %s\n", k, config[k])
	}
	if len(config) > 0 {
		b.WriteString("\n")
	}
	for _, u := range benchUnits {
		fmt.Fprintf(&b, "Unit %s better=lower assume=nothing\n", u)
	}
	for _, r := range rows {
		iters := r.Iterations
		if iters <= 0 {
			iters = 1
		}
		fmt.Fprintf(&b, "%s %d %s %s %s %s %s %s %s %s %s %s\n",
			benchName(r.Name), iters,
			formatValue(r.PeakRSSBytes), "peak-RSS-bytes",
			formatValue(r.SteadyRSSBytes), "steady-RSS-bytes",
			formatValue(r.GCLiveBytes), "gc-live-bytes",
			formatValue(r.GoroutinesSteady), "goroutines-steady",
			formatValue(r.P99LatencyNs), "p99-latency-ns",
		)
	}
	return b.String()
}

func benchName(name string) string {
	name = strings.ReplaceAll(strings.TrimSpace(name), " ", "_")
	if !strings.HasPrefix(name, "Benchmark") {
		name = "Benchmark" + name
	}
	return name
}

func formatValue(v float64) string {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return "0"
	}
	if v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return strconv.FormatInt(int64(v), 10)
	}
	return strconv.FormatFloat(v, 'f', 3, 64)
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j] < keys[j-1]; j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
	return keys
}
