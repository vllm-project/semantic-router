package benchmark

import (
	"bufio"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
)

// procSuffix matches the trailing "-<GOMAXPROCS>" the Go test runner appends to
// every benchmark line (e.g. "BenchmarkFoo/case-1-8"). Only this final segment
// is stripped, so slashes and dashes inside a subtest name are preserved.
var procSuffix = regexp.MustCompile(`-\d+$`)

// ParseBenchOutput converts raw `go test -bench` output into a Baseline. It is
// the producer of the "current" result set that #2455 root cause #2 was
// missing: nothing turned live benchmark output into the current.json the
// comparison path needs.
//
// A benchmark line looks like:
//
//	BenchmarkName-8   \t 60389019 \t 628.5 ns/op \t 112 B/op \t 5 allocs/op
//
// ns/op is required; the B/op and allocs/op columns are optional (absent
// without -benchmem). Non-benchmark lines (goos:, PASS, ok, …) are ignored.
func ParseBenchOutput(r io.Reader) (*Baseline, error) {
	baseline := &Baseline{Benchmarks: make(map[string]BenchmarkMetric)}

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 4 || !strings.HasPrefix(fields[0], "Benchmark") {
			continue
		}
		metric, ok := parseMetricColumns(fields[1:])
		if !ok {
			continue
		}
		name := procSuffix.ReplaceAllString(fields[0], "")
		baseline.Benchmarks[name] = metric
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to read benchmark output: %w", err)
	}
	return baseline, nil
}

// parseMetricColumns reads the "<value> <unit>" pairs that follow the benchmark
// name. It returns ok=false when no ns/op column is present, so malformed or
// non-result lines are skipped rather than recorded as zero-valued metrics.
func parseMetricColumns(cols []string) (BenchmarkMetric, bool) {
	var metric BenchmarkMetric
	haveNs := false
	for i, tok := range cols {
		if i == 0 {
			continue // iterations count; not retained
		}
		switch tok {
		case "ns/op":
			if v, err := strconv.ParseFloat(cols[i-1], 64); err == nil {
				metric.NsPerOp = v
				haveNs = true
			}
		case "B/op":
			if v, err := strconv.ParseInt(cols[i-1], 10, 64); err == nil {
				metric.BytesPerOp = v
			}
		case "allocs/op":
			if v, err := strconv.ParseInt(cols[i-1], 10, 64); err == nil {
				metric.AllocsPerOp = v
			}
		}
	}
	return metric, haveNs
}
