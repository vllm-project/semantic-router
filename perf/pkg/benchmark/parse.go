package benchmark

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

var procSuffix = regexp.MustCompile(`-\d+$`)

// ParseBenchOutput parses Go benchmark output and aggregates repeated -count
// samples by median. NsStdDevPct exposes noisy runs in reports without turning
// host-dependent timing into a portable gate.
func ParseBenchOutput(reader io.Reader) (*Baseline, error) {
	return ParseBenchOutputForSuite(reader, "")
}

func ParseBenchOutputForSuite(reader io.Reader, suite string) (*Baseline, error) {
	samples := make(map[string][]BenchmarkMetric)
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		name, metric, ok := parseBenchmarkLine(scanner.Text())
		if !ok {
			continue
		}
		metric.Suite = suite
		samples[name] = append(samples[name], metric)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read benchmark output: %w", err)
	}

	result := &Baseline{SchemaVersion: ResultSchemaVersion, Benchmarks: make(map[string]BenchmarkMetric, len(samples))}
	for name, values := range samples {
		result.Benchmarks[name] = aggregateMetrics(values)
	}
	return result, nil
}

func parseBenchmarkLine(line string) (string, BenchmarkMetric, bool) {
	fields := strings.Fields(line)
	if len(fields) < 4 || !strings.HasPrefix(fields[0], "Benchmark") {
		return "", BenchmarkMetric{}, false
	}
	iterations, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil {
		return "", BenchmarkMetric{}, false
	}
	metric := BenchmarkMetric{Iterations: iterations, Custom: make(map[string]float64)}
	haveNs := false
	for i := 2; i+1 < len(fields); i += 2 {
		value, err := strconv.ParseFloat(fields[i], 64)
		if err != nil {
			continue
		}
		unit := fields[i+1]
		switch unit {
		case "ns/op":
			metric.NsPerOp = value
			haveNs = true
		case "B/op":
			metric.BytesPerOp = int64(math.Round(value))
		case "allocs/op":
			metric.AllocsPerOp = int64(math.Round(value))
		default:
			metric.Custom[unit] = value
		}
	}
	if !haveNs {
		return "", BenchmarkMetric{}, false
	}
	if len(metric.Custom) == 0 {
		metric.Custom = nil
	}
	return procSuffix.ReplaceAllString(fields[0], ""), metric, true
}

func aggregateMetrics(samples []BenchmarkMetric) BenchmarkMetric {
	if len(samples) == 0 {
		return BenchmarkMetric{}
	}
	nsValues := make([]float64, 0, len(samples))
	iterationValues := make([]int64, 0, len(samples))
	allocationValues := make([]int64, 0, len(samples))
	byteValues := make([]int64, 0, len(samples))
	customValues := make(map[string][]float64)
	for _, sample := range samples {
		nsValues = append(nsValues, sample.NsPerOp)
		iterationValues = append(iterationValues, sample.Iterations)
		allocationValues = append(allocationValues, sample.AllocsPerOp)
		byteValues = append(byteValues, sample.BytesPerOp)
		for unit, value := range sample.Custom {
			customValues[unit] = append(customValues[unit], value)
		}
	}
	metric := BenchmarkMetric{
		Suite:       samples[0].Suite,
		Iterations:  medianInt(iterationValues),
		Samples:     len(samples),
		NsPerOp:     medianFloat(nsValues),
		NsStdDevPct: coefficientOfVariation(nsValues),
		AllocsPerOp: medianInt(allocationValues),
		BytesPerOp:  medianInt(byteValues),
		Custom:      make(map[string]float64, len(customValues)),
	}
	for unit, values := range customValues {
		metric.Custom[unit] = medianFloat(values)
	}
	if len(metric.Custom) == 0 {
		metric.Custom = nil
	}
	return metric
}

func medianFloat(values []float64) float64 {
	ordered := append([]float64(nil), values...)
	sort.Float64s(ordered)
	middle := len(ordered) / 2
	if len(ordered)%2 == 0 {
		return (ordered[middle-1] + ordered[middle]) / 2
	}
	return ordered[middle]
}

func medianInt(values []int64) int64 {
	ordered := append([]int64(nil), values...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	middle := len(ordered) / 2
	if len(ordered)%2 == 0 {
		return int64(math.Round(float64(ordered[middle-1]+ordered[middle]) / 2))
	}
	return ordered[middle]
}

func coefficientOfVariation(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}
	mean := 0.0
	for _, value := range values {
		mean += value
	}
	mean /= float64(len(values))
	if mean == 0 {
		return 0
	}
	variance := 0.0
	for _, value := range values {
		delta := value - mean
		variance += delta * delta
	}
	variance /= float64(len(values) - 1)
	return math.Sqrt(variance) / mean * 100
}
