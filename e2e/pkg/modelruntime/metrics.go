package modelruntime

import (
	"bufio"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Router metrics of model_runtime deployments (pkg/modelservice).
const (
	RouterReadyMetric     = "vsr_model_runtime_ready"
	RouterRestartsMetric  = "vsr_model_runtime_restarts_total"
	RouterRequestsMetric  = "vsr_model_runtime_requests_total"
	RouterUnknownMetric   = "vsr_model_runtime_unknown_answers_total"
	RouterTransportMetric = "vsr_model_runtime_transport_seconds"
	RouterServerMetric    = "vsr_model_runtime_server_seconds"
)

// Sample is one Prometheus sample.
type Sample struct {
	Name   string
	Labels map[string]string
	Value  float64
}

// Metrics is a parsed Prometheus text exposition.
type Metrics struct {
	Samples []Sample
}

// ParseMetrics parses the Prometheus text format, ignoring comments.
func ParseMetrics(text string) (Metrics, error) {
	var metrics Metrics
	scanner := bufio.NewScanner(strings.NewReader(text))
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		sample, err := parseSample(line)
		if err != nil {
			return metrics, err
		}
		metrics.Samples = append(metrics.Samples, sample)
	}
	return metrics, scanner.Err()
}

func parseSample(line string) (Sample, error) {
	sample := Sample{Labels: map[string]string{}}
	var rest string
	if open := strings.IndexByte(line, '{'); open >= 0 {
		end := strings.LastIndexByte(line, '}')
		if end < open {
			return sample, fmt.Errorf("malformed sample %q", line)
		}
		sample.Name = line[:open]
		labels, err := parseLabels(line[open+1 : end])
		if err != nil {
			return sample, fmt.Errorf("sample %q: %w", line, err)
		}
		sample.Labels = labels
		rest = strings.TrimSpace(line[end+1:])
	} else {
		name, value, found := strings.Cut(line, " ")
		if !found {
			return sample, fmt.Errorf("malformed sample %q", line)
		}
		sample.Name, rest = name, strings.TrimSpace(value)
	}
	fields := strings.Fields(rest)
	if len(fields) == 0 {
		return sample, fmt.Errorf("sample %q has no value", line)
	}
	value, err := parsePrometheusFloat(fields[0])
	if err != nil {
		return sample, fmt.Errorf("sample %q: %w", line, err)
	}
	sample.Value = value
	return sample, nil
}

func parseLabels(text string) (map[string]string, error) {
	labels := map[string]string{}
	for len(strings.TrimSpace(text)) > 0 {
		text = strings.TrimLeft(text, " ,")
		name, rest, found := strings.Cut(text, "=")
		if !found || !strings.HasPrefix(rest, `"`) {
			return nil, fmt.Errorf("malformed labels %q", text)
		}
		var value strings.Builder
		index := 1
		for ; index < len(rest); index++ {
			char := rest[index]
			if char == '\\' && index+1 < len(rest) {
				index++
				switch rest[index] {
				case 'n':
					value.WriteByte('\n')
				default:
					value.WriteByte(rest[index])
				}
				continue
			}
			if char == '"' {
				break
			}
			value.WriteByte(char)
		}
		if index >= len(rest) {
			return nil, fmt.Errorf("unterminated label value in %q", text)
		}
		labels[strings.TrimSpace(name)] = value.String()
		text = rest[index+1:]
	}
	return labels, nil
}

func parsePrometheusFloat(value string) (float64, error) {
	switch value {
	case "+Inf":
		return math.Inf(1), nil
	case "-Inf":
		return math.Inf(-1), nil
	case "NaN":
		return math.NaN(), nil
	}
	return strconv.ParseFloat(value, 64)
}

// Value returns the first sample of name whose labels include match.
func (m Metrics) Value(name string, match map[string]string) (float64, bool) {
	for _, sample := range m.Samples {
		if sample.Name == name && labelsInclude(sample.Labels, match) {
			return sample.Value, true
		}
	}
	return 0, false
}

// Sum adds every sample of name whose labels include match.
func (m Metrics) Sum(name string, match map[string]string) float64 {
	total := 0.0
	for _, sample := range m.Samples {
		if sample.Name == name && labelsInclude(sample.Labels, match) {
			total += sample.Value
		}
	}
	return total
}

func labelsInclude(labels, match map[string]string) bool {
	for key, want := range match {
		if labels[key] != want {
			return false
		}
	}
	return true
}

// DeploymentReady reports the Router's readiness gauge for one deployment.
func (m Metrics) DeploymentReady(deployment string) bool {
	value, ok := m.Value(RouterReadyMetric, map[string]string{"deployment": deployment})
	return ok && value == 1
}

// DeploymentRestarts reports how often the Router restarted a deployment's process.
func (m Metrics) DeploymentRestarts(deployment string) float64 {
	return m.Sum(RouterRestartsMetric, map[string]string{"deployment": deployment})
}
