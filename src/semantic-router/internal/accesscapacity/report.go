package accesscapacity

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const reportSchema = "vllm-sr.access-capacity-report/v1"

type Report struct {
	Schema      string      `json:"schema"`
	Status      string      `json:"status"`
	StartedAt   time.Time   `json:"startedAt"`
	FinishedAt  time.Time   `json:"finishedAt"`
	DurationMS  float64     `json:"durationMs"`
	Parameters  Parameters  `json:"parameters"`
	Environment Environment `json:"environment"`
	Projection  Projection  `json:"projection"`
	Admission   Admission   `json:"admission"`
	Usage       Usage       `json:"usage"`
	Failover    Failover    `json:"failover"`
	Checks      []Check     `json:"checks"`
	Errors      []string    `json:"errors,omitempty"`
}

type Parameters struct {
	KeyCount     int `json:"keyCount"`
	Replicas     int `json:"replicas"`
	Concurrency  int `json:"concurrency"`
	RequestLimit int `json:"requestLimit"`
}

type Environment struct {
	GoVersion    string `json:"goVersion"`
	RedisVersion string `json:"redisVersion,omitempty"`
	RedisMode    string `json:"redisMode,omitempty"`
	Transport    string `json:"transport"`
}

type Projection struct {
	FixtureBuildMS      float64        `json:"fixtureBuildMs"`
	CompileMS           float64        `json:"compileMs"`
	PublishMS           float64        `json:"publishMs"`
	TotalMS             float64        `json:"totalMs"`
	KeysPerSecond       float64        `json:"keysPerSecond"`
	RedisKeyCount       int64          `json:"redisKeyCount"`
	MemoryBytes         int64          `json:"memoryBytes"`
	MemoryBytesPerKey   float64        `json:"memoryBytesPerKey"`
	RedisOps            RedisOperation `json:"redisOps"`
	RedisOpsPerKey      float64        `json:"redisOpsPerKey"`
	VisibilitySets      int            `json:"visibilitySets"`
	IsolationSamples    int            `json:"isolationSamples"`
	IsolationViolations int            `json:"isolationViolations"`
}

type Admission struct {
	Attempted           int64          `json:"attempted"`
	Allowed             int64          `json:"allowed"`
	RateLimited         int64          `json:"rateLimited"`
	Failed              int64          `json:"failed"`
	EventsPerSecond     float64        `json:"eventsPerSecond"`
	Authentication      Latency        `json:"authenticationLatency"`
	Admission           Latency        `json:"admissionLatency"`
	Settlement          Latency        `json:"settlementLatency"`
	RedisOps            RedisOperation `json:"redisOps"`
	RedisOpsPerEvent    float64        `json:"redisOpsPerEvent"`
	MemoryDeltaBytes    int64          `json:"memoryDeltaBytes"`
	MemoryBytesPerEvent float64        `json:"memoryBytesPerEvent"`
}

type Usage struct {
	Produced        int64   `json:"produced"`
	Observed        int64   `json:"observed"`
	Acknowledged    int64   `json:"acknowledged"`
	RetainedEntries int64   `json:"retainedEntries"`
	PendingEntries  int64   `json:"pendingEntries"`
	GroupLag        int64   `json:"groupLag"`
	ObservationLag  Latency `json:"observationLag"`
}

type Failover struct {
	Scope                        string  `json:"scope"`
	ReplicasBefore               int     `json:"replicasBefore"`
	ReplicasAfter                int     `json:"replicasAfter"`
	FailedReplicaRequestRejected bool    `json:"failedReplicaRequestRejected"`
	ReroutedRequestAllowed       bool    `json:"reroutedRequestAllowed"`
	ReplacementRequestAllowed    bool    `json:"replacementRequestAllowed"`
	PostLimitRequestDenied       bool    `json:"postLimitRequestDenied"`
	ExpectedAllowedForSentinel   int     `json:"expectedAllowedForSentinel"`
	ObservedAllowedForSentinel   int     `json:"observedAllowedForSentinel"`
	TransitionMS                 float64 `json:"transitionMs"`
	Errors                       int     `json:"errors"`
	GlobalQuotaStateConsistent   bool    `json:"globalQuotaStateConsistent"`
}

type RedisOperation struct {
	Total     int64            `json:"total"`
	ByCommand map[string]int64 `json:"byCommand"`
}

type Latency struct {
	Count int     `json:"count"`
	P50MS float64 `json:"p50Ms"`
	P95MS float64 `json:"p95Ms"`
	P99MS float64 `json:"p99Ms"`
	MaxMS float64 `json:"maxMs"`
}

type Check struct {
	Name      string `json:"name"`
	Status    string `json:"status"`
	Observed  string `json:"observed"`
	Threshold string `json:"threshold"`
}

func NewReport(config Config, startedAt time.Time, transport string) Report {
	environment := workloadEnvironment()
	environment.Transport = transport
	return Report{
		Schema: reportSchema, Status: "failed", StartedAt: startedAt.UTC(),
		Parameters: Parameters{
			KeyCount: config.KeyCount, Replicas: config.Replicas, Concurrency: config.Concurrency,
			RequestLimit: config.RequestLimit,
		},
		Environment: environment,
		Failover:    Failover{Scope: "router_replica", ReplicasBefore: config.Replicas},
	}
}

func (r *Report) Complete(config Config) {
	r.FinishedAt = time.Now().UTC()
	r.DurationMS = milliseconds(r.FinishedAt.Sub(r.StartedAt))
	r.Checks = []Check{
		check("10,000-key projection", r.Parameters.KeyCount >= DefaultKeyCount,
			fmt.Sprintf("%d keys", r.Parameters.KeyCount), fmt.Sprintf(">= %d keys", DefaultKeyCount)),
		check("projection throughput", r.Projection.KeysPerSecond >= config.Thresholds.MinProjectionKeysPerS,
			fmt.Sprintf("%.2f keys/s", r.Projection.KeysPerSecond), fmt.Sprintf(">= %.2f keys/s", config.Thresholds.MinProjectionKeysPerS)),
		check("projection memory", r.Projection.MemoryBytesPerKey <= float64(config.Thresholds.MaxProjectionBytesKey),
			fmt.Sprintf("%.2f bytes/key", r.Projection.MemoryBytesPerKey), fmt.Sprintf("<= %d bytes/key", config.Thresholds.MaxProjectionBytesKey)),
		check("policy isolation", r.Projection.IsolationSamples > 0 && r.Projection.IsolationViolations == 0,
			fmt.Sprintf("%d violations in %d samples", r.Projection.IsolationViolations, r.Projection.IsolationSamples), "0 violations"),
		check("admission completion", r.Admission.Attempted == int64(config.KeyCount) && r.Admission.Allowed == int64(config.KeyCount) && r.Admission.Failed == 0,
			fmt.Sprintf("%d/%d allowed, %d failed", r.Admission.Allowed, r.Admission.Attempted, r.Admission.Failed), "all keys admitted once"),
		check("admission p99", r.Admission.Admission.P99MS <= milliseconds(config.Thresholds.MaxAdmissionP99),
			fmt.Sprintf("%.3f ms", r.Admission.Admission.P99MS), fmt.Sprintf("<= %.3f ms", milliseconds(config.Thresholds.MaxAdmissionP99))),
		check("event memory", r.Admission.MemoryBytesPerEvent <= float64(config.Thresholds.MaxEventBytes),
			fmt.Sprintf("%.2f bytes/event", r.Admission.MemoryBytesPerEvent), fmt.Sprintf("<= %d bytes/event", config.Thresholds.MaxEventBytes)),
		check("usage delivery", r.Usage.Produced == r.Usage.Observed && r.Usage.Observed == r.Usage.Acknowledged && r.Usage.PendingEntries == 0 && r.Usage.GroupLag == 0,
			fmt.Sprintf("produced=%d observed=%d acked=%d pending=%d lag=%d", r.Usage.Produced, r.Usage.Observed, r.Usage.Acknowledged, r.Usage.PendingEntries, r.Usage.GroupLag), "no lost or pending events"),
		check("usage p99 lag", r.Usage.ObservationLag.P99MS <= milliseconds(config.Thresholds.MaxUsageLagP99),
			fmt.Sprintf("%.3f ms", r.Usage.ObservationLag.P99MS), fmt.Sprintf("<= %.3f ms", milliseconds(config.Thresholds.MaxUsageLagP99))),
		check("replica failover", r.Failover.GlobalQuotaStateConsistent && r.Failover.Errors == 0,
			fmt.Sprintf("quota consistent=%t, errors=%d", r.Failover.GlobalQuotaStateConsistent, r.Failover.Errors), "global quota preserved"),
	}
	r.Status = "passed"
	for _, item := range r.Checks {
		if item.Status != "passed" {
			r.Status = "failed"
			break
		}
	}
	if len(r.Errors) != 0 {
		r.Status = "failed"
	}
}

func check(name string, passed bool, observed, threshold string) Check {
	status := "failed"
	if passed {
		status = "passed"
	}
	return Check{Name: name, Status: status, Observed: observed, Threshold: threshold}
}

func appendReportError(report *Report, err error) {
	if err == nil {
		return
	}
	message := strings.TrimSpace(err.Error())
	if firstLine, _, found := strings.Cut(message, "\n"); found {
		message = firstLine
	}
	if operation, _, found := strings.Cut(message, ": "); found {
		message = operation
	}
	if message == "" {
		message = "capacity gate operation failed"
	}
	report.Errors = append(report.Errors, message)
}

func latency(values []time.Duration) Latency {
	if len(values) == 0 {
		return Latency{}
	}
	sorted := append([]time.Duration(nil), values...)
	sort.Slice(sorted, func(left, right int) bool { return sorted[left] < sorted[right] })
	return Latency{
		Count: len(sorted), P50MS: milliseconds(percentile(sorted, 0.50)),
		P95MS: milliseconds(percentile(sorted, 0.95)), P99MS: milliseconds(percentile(sorted, 0.99)),
		MaxMS: milliseconds(sorted[len(sorted)-1]),
	}
}

func percentile(sorted []time.Duration, quantile float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	index := int(math.Ceil(quantile*float64(len(sorted)))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func milliseconds(value time.Duration) float64 {
	return float64(value) / float64(time.Millisecond)
}

func WriteReport(root string, report Report) (string, error) {
	runID := report.StartedAt.UTC().Format("20060102T150405.000000000Z")
	directory := filepath.Join(root, runID)
	if err := os.MkdirAll(directory, 0o750); err != nil {
		return "", fmt.Errorf("create report directory: %w", err)
	}
	payload, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return "", fmt.Errorf("encode report: %w", err)
	}
	payload = append(payload, '\n')
	if err := os.WriteFile(filepath.Join(directory, "report.json"), payload, 0o600); err != nil {
		return "", fmt.Errorf("write JSON report: %w", err)
	}
	if err := os.WriteFile(filepath.Join(directory, "summary.md"), []byte(report.Markdown()), 0o600); err != nil {
		return "", fmt.Errorf("write Markdown report: %w", err)
	}
	return directory, nil
}

func (r Report) Markdown() string {
	var output strings.Builder
	fmt.Fprintf(&output, "# Access-control capacity gate\n\n")
	fmt.Fprintf(&output, "**Result:** %s  \n", strings.ToUpper(r.Status))
	fmt.Fprintf(&output, "**Workload:** %s keys · %d replicas · %d concurrent workers\n\n",
		comma(r.Parameters.KeyCount), r.Parameters.Replicas, r.Parameters.Concurrency)
	fmt.Fprintf(&output, "**Scope:** `router_replica` — production runtime components against Valkey; not Router/Envoy HTTP E2E.\n\n")
	fmt.Fprintf(&output, "| Signal | Result |\n| --- | ---: |\n")
	fmt.Fprintf(&output, "| Projection throughput | %.2f keys/s |\n", r.Projection.KeysPerSecond)
	fmt.Fprintf(&output, "| Projection memory | %.2f bytes/key |\n", r.Projection.MemoryBytesPerKey)
	fmt.Fprintf(&output, "| Admission p50 / p95 / p99 | %.3f / %.3f / %.3f ms |\n",
		r.Admission.Admission.P50MS, r.Admission.Admission.P95MS, r.Admission.Admission.P99MS)
	fmt.Fprintf(&output, "| Redis operations | %s (%.2f/key) projection · %s (%.2f/event) workload |\n",
		comma64(r.Projection.RedisOps.Total), r.Projection.RedisOpsPerKey,
		comma64(r.Admission.RedisOps.Total), r.Admission.RedisOpsPerEvent)
	fmt.Fprintf(&output, "| Event memory | %.2f bytes/event |\n", r.Admission.MemoryBytesPerEvent)
	fmt.Fprintf(&output, "| Usage observation p99 | %.3f ms |\n", r.Usage.ObservationLag.P99MS)
	fmt.Fprintf(&output, "| Replica failover | consistent=%t · %.3f ms transition |\n\n",
		r.Failover.GlobalQuotaStateConsistent, r.Failover.TransitionMS)
	fmt.Fprintf(&output, "## Checks\n\n| Check | Status | Observed | Gate |\n| --- | --- | --- | --- |\n")
	for _, item := range r.Checks {
		fmt.Fprintf(&output, "| %s | %s | %s | %s |\n", item.Name, item.Status, item.Observed, item.Threshold)
	}
	if len(r.Errors) != 0 {
		fmt.Fprintf(&output, "\n## Errors\n\n")
		for _, item := range r.Errors {
			fmt.Fprintf(&output, "- %s\n", item)
		}
	}
	return output.String()
}

func comma(value int) string { return comma64(int64(value)) }

func comma64(value int64) string {
	text := fmt.Sprintf("%d", value)
	for offset := len(text) - 3; offset > 0; offset -= 3 {
		text = text[:offset] + "," + text[offset:]
	}
	return text
}
