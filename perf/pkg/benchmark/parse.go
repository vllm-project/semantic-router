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
// without -benchmem). A benchmark may print progress between the Go runner's
// name and its numeric result. Retain that name until its result arrives;
// reject incomplete or ambiguous measurements instead of silently losing them.
func ParseBenchOutput(r io.Reader) (*Baseline, error) {
	baseline := &Baseline{Benchmarks: make(map[string]BenchmarkMetric)}
	identities := map[string]*ModelIdentity{}
	pending := ""

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, ModelIdentityPrefix) {
			name, identity, err := parseModelIdentity(line)
			if err != nil {
				return nil, err
			}
			identities[name] = identity
			continue
		}
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		name := pending
		columns := fields
		hasName := strings.HasPrefix(fields[0], "Benchmark")
		if hasName {
			if pending != "" {
				return nil, fmt.Errorf("benchmark %s has no numeric result before %s", pending, fields[0])
			}
			name, columns = fields[0], fields[1:]
			if len(columns) == 0 {
				pending = name
				continue
			}
		} else {
			if pending != "" && benchmarkOutputBoundary(line) {
				return nil, fmt.Errorf("benchmark %s has no numeric result before %s", pending, fields[0])
			}
			if _, err := strconv.ParseUint(fields[0], 10, 64); err != nil {
				continue
			}
		}
		metric, ok := parseMetricColumns(columns)
		if !ok {
			if hasName {
				pending = name
			}
			continue
		}
		iterations, err := strconv.ParseUint(columns[0], 10, 64)
		if err != nil || iterations == 0 {
			return nil, fmt.Errorf("benchmark %s has invalid iteration count %q", name, columns[0])
		}
		if name == "" {
			return nil, fmt.Errorf("numeric benchmark result has no benchmark name: %s", line)
		}
		name = procSuffix.ReplaceAllString(name, "")
		if _, exists := baseline.Benchmarks[name]; exists {
			return nil, fmt.Errorf("duplicate benchmark result %s; compare one measurement per workload", name)
		}
		baseline.Benchmarks[name] = metric
		pending = ""
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to read benchmark output: %w", err)
	}
	if pending != "" {
		return nil, fmt.Errorf("benchmark %s has no numeric result before end of output", pending)
	}
	for name, metric := range baseline.Benchmarks {
		metric.ModelIdentity = identities[name]
		baseline.Benchmarks[name] = metric
	}
	return baseline, nil
}

func benchmarkOutputBoundary(line string) bool {
	for _, prefix := range []string{"PASS", "FAIL", "ok\t", "ok ", "?", "goos:", "goarch:", "pkg:", "--- FAIL:", "panic:"} {
		if strings.HasPrefix(line, prefix) {
			return true
		}
	}
	return false
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
