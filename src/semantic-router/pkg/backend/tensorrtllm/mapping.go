package tensorrtllm

import (
	"math"
	"sort"
	"time"

	dto "github.com/prometheus/client_model/go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backend"
)

// TensorRT-LLM batch-manager gauge families. TRT-LLM/Triton pack multiple
// signals into a single gauge family disambiguated by a label dimension, unlike
// vLLM which uses one distinct metric name per signal.
const (
	MetricRequestMetrics       = "nv_trt_llm_request_metrics"
	MetricKVCacheBlockMetrics  = "nv_trt_llm_kv_cache_block_metrics"
	MetricRuntimeMemoryMetrics = "nv_trt_llm_runtime_memory_metrics"
	MetricInflightBatcher      = "nv_trt_llm_inflight_batcher_metrics"
	MetricDisaggregatedServing = "nv_trt_llm_disaggregated_serving_metrics"
	MetricGeneralMetrics       = "nv_trt_llm_general_metrics"

	// Label keys used by the TRT-LLM gauge families.
	LabelRequestType       = "request_type"
	LabelKVCacheBlockType  = "kv_cache_block_type"
	LabelMemoryType        = "memory_type"
	LabelInflightBatcher   = "inflight_batcher_specific_metric"
	LabelDisaggregatedType = "disaggregated_serving_type"

	// Base Triton core metrics (always present).
	MetricPendingRequestCount = "nv_inference_pending_request_count"
	MetricRequestDurationUs   = "nv_inference_request_duration_us"
	MetricQueueDurationUs     = "nv_inference_queue_duration_us"
	MetricRequestSuccess      = "nv_inference_request_success"
	MetricGPUUtilization      = "nv_gpu_utilization"
	MetricGPUMemoryUsed       = "nv_gpu_memory_used_bytes"
	MetricGPUMemoryTotal      = "nv_gpu_memory_total_bytes"

	// Experimental latency families (opt-in via --metrics-config). Treated as
	// opportunistic enrichment only; the adapter never depends on them.
	MetricFirstResponseHistogramMs = "nv_inference_first_response_histogram_ms"
	MetricRequestSummaryUs         = "nv_inference_request_summary_us"
	MetricQueueSummaryUs           = "nv_inference_queue_summary_us"
)

// counterSnapshot holds the raw cumulative counters used to derive average
// latency across two consecutive scrapes (Triton core exposes counters, not
// histograms, by default).
type counterSnapshot struct {
	requestDurationUs float64
	queueDurationUs   float64
	requestCount      float64
	valid             bool
}

// normalizeTarget converts one scraped Triton/TRT-LLM metrics document into a
// single BackendTelemetry sample. Because Triton is one-process-one-endpoint,
// one scrape target maps to exactly one replica (naive 1:1 identity).
//
// prev is the previous counter snapshot for this target (zero value if none);
// the returned counterSnapshot should be stored for the next scrape's latency
// delta. recognized reports whether any TRT-LLM or Triton metric was found.
func normalizeTarget(
	target backend.AdapterTarget,
	families map[string]*dto.MetricFamily,
	prev counterSnapshot,
	ttl time.Duration,
	now time.Time,
) (backend.BackendTelemetry, counterSnapshot, bool) {
	identity := target.Identity.Normalize()
	identity.EngineKind = backend.EngineKindTensorRTLLM

	t := backend.BackendTelemetry{
		Identity:    identity,
		Health:      backend.HealthStateHealthy,
		Confidence:  1,
		CollectedAt: now,
		TTL:         ttl,
	}

	sig := signals{}
	sig.merge(applyRequestMetrics(&t, families))
	sig.merge(applyKVCacheMetrics(&t, families))
	sig.merge(applyGPUMetrics(&t, families))
	sig.merge(applyDisaggregatedMetrics(&t, families))

	// Latency, highest available fidelity tier (see design 4.1).
	snapshot := applyLatency(&t, families, prev)
	if snapshot.valid {
		sig.any = true
	}

	// Health derivation: degrade when saturated. Never assert unhealthy from a
	// single scrape (staleness in the store is the unhealthy path).
	if isSaturated(&t) {
		t.Health = backend.HealthStateDegraded
	}

	// Confidence: lower when only base Triton metrics were present (no
	// engine-specific nv_trt_llm_* signals), so policy can prefer richer backends.
	if !sig.trtLLM {
		t.Confidence = 0.5
	}

	return t, snapshot, sig.any
}

// signals tracks whether a metric group produced any recognized signal and
// whether an engine-specific (nv_trt_llm_*) signal was present.
type signals struct {
	any    bool
	trtLLM bool
}

func (s *signals) merge(other signals) {
	s.any = s.any || other.any
	s.trtLLM = s.trtLLM || other.trtLLM
}

// applyRequestMetrics maps queue depth and active requests. Queue depth prefers
// the TRT-LLM waiting gauge and falls back to the Triton pending counter.
func applyRequestMetrics(t *backend.BackendTelemetry, families map[string]*dto.MetricFamily) signals {
	sig := signals{}
	if v, ok := labeledGauge(families, MetricRequestMetrics, LabelRequestType, "waiting"); ok {
		queue := int(math.Round(v))
		t.QueueDepth = &queue
		sig.trtLLM = true
		sig.any = true
	} else if v, ok := plainGauge(families, MetricPendingRequestCount); ok {
		queue := int(math.Round(v))
		t.QueueDepth = &queue
		sig.any = true
	}
	if v, ok := labeledGauge(families, MetricRequestMetrics, LabelRequestType, "active"); ok {
		active := int(math.Round(v))
		t.ActiveRequests = &active
		sig.trtLLM = true
		sig.any = true
	}
	return sig
}

// applyKVCacheMetrics maps KV cache pressure (fraction) and block-reuse headroom.
func applyKVCacheMetrics(t *backend.BackendTelemetry, families map[string]*dto.MetricFamily) signals {
	sig := signals{}
	if frac, ok := labeledGauge(families, MetricKVCacheBlockMetrics, LabelKVCacheBlockType, "fraction"); ok {
		clamped := clamp01(frac)
		t.KVCachePressure = &clamped
		sig.trtLLM = true
		sig.any = true
	}
	used, usedOK := labeledGauge(families, MetricKVCacheBlockMetrics, LabelKVCacheBlockType, "used")
	maxBlocks, maxOK := labeledGauge(families, MetricKVCacheBlockMetrics, LabelKVCacheBlockType, "max")
	if usedOK && maxOK && maxBlocks > 0 {
		reuse := clamp01(1 - used/maxBlocks)
		t.Affinity.KVCacheReuseScore = &reuse
		sig.trtLLM = true
		sig.any = true
	}
	return sig
}

// applyGPUMetrics maps GPU utilization (max across GPUs) and memory pressure
// (aggregate used/total across GPUs). These are base Triton metrics.
func applyGPUMetrics(t *backend.BackendTelemetry, families map[string]*dto.MetricFamily) signals {
	sig := signals{}
	if util, ok := gpuUtilizationMax(families); ok {
		t.GPUUtilization = &util
		sig.any = true
	}
	if mem, ok := gpuMemoryPressure(families); ok {
		t.MemoryPressure = &mem
		sig.any = true
	}
	return sig
}

// applyDisaggregatedMetrics maps the optional P/D disaggregation KV-transfer hint.
func applyDisaggregatedMetrics(t *backend.BackendTelemetry, families map[string]*dto.MetricFamily) signals {
	sig := signals{}
	if v, ok := labeledMetricValue(families, MetricDisaggregatedServing, LabelDisaggregatedType, "kv_cache_transfer_ms"); ok {
		if t.Affinity.ExtraHints == nil {
			t.Affinity.ExtraHints = map[string]float64{}
		}
		t.Affinity.ExtraHints["kv_cache_transfer_ms"] = v
		sig.trtLLM = true
		sig.any = true
	}
	return sig
}

// applyLatency fills latency using the highest-fidelity source available.
// Priority: latency summaries (quantiles) > TTFT histogram (percentiles) >
// counter deltas (average only). Returns the counter snapshot for next scrape.
func applyLatency(t *backend.BackendTelemetry, families map[string]*dto.MetricFamily, prev counterSnapshot) counterSnapshot {
	// Tier 3: experimental summaries (configurable quantiles).
	if snap, ok := summarySnapshot(families, MetricRequestSummaryUs, 1e6); ok {
		t.Latency.E2ESeconds = snap
	}
	if snap, ok := summarySnapshot(families, MetricQueueSummaryUs, 1e6); ok {
		t.Latency.QueueSeconds = snap
	}

	// Tier 2: experimental TTFT histogram (ms -> s).
	if snap, ok := histogramSnapshot(families, MetricFirstResponseHistogramMs, 1e3); ok {
		t.Latency.TTFTSeconds = snap
	}

	// Tier 1: always-on counter averages (E2E/queue). Only fills percentile-less
	// average into P50 as a coarse central estimate when no richer tier set it.
	current := readCounters(families)
	if current.valid && prev.valid {
		fillAverageIfEmpty(&t.Latency.E2ESeconds, current.requestDurationUs, prev.requestDurationUs, current.requestCount, prev.requestCount)
		fillAverageIfEmpty(&t.Latency.QueueSeconds, current.queueDurationUs, prev.queueDurationUs, current.requestCount, prev.requestCount)
	}
	return current
}

// fillAverageIfEmpty sets the snapshot's P50 to the counter-delta average when a
// richer latency tier has not already populated it.
func fillAverageIfEmpty(snap *backend.LatencySnapshot, durNow, durPrev, countNow, countPrev float64) {
	if snap.P50Seconds != nil {
		return
	}
	if avg, ok := counterDeltaAvgSeconds(durNow, durPrev, countNow, countPrev); ok {
		snap.P50Seconds = &avg
	}
}

func isSaturated(t *backend.BackendTelemetry) bool {
	waiting := t.QueueDepth != nil && *t.QueueDepth > 0
	kvFull := t.KVCachePressure != nil && *t.KVCachePressure >= 0.95
	return waiting && kvFull
}

// --- Prometheus family readers ---

// labeledGauge returns the value of a gauge family metric whose label labelKey
// equals labelVal.
func labeledGauge(families map[string]*dto.MetricFamily, name, labelKey, labelVal string) (float64, bool) {
	family := families[name]
	if family == nil {
		return 0, false
	}
	for _, m := range family.Metric {
		if labelValue(m, labelKey) != labelVal {
			continue
		}
		if g := m.GetGauge(); g != nil {
			return g.GetValue(), true
		}
	}
	return 0, false
}

// labeledMetricValue reads a gauge or counter value for a labeled metric.
func labeledMetricValue(families map[string]*dto.MetricFamily, name, labelKey, labelVal string) (float64, bool) {
	family := families[name]
	if family == nil {
		return 0, false
	}
	for _, m := range family.Metric {
		if labelValue(m, labelKey) != labelVal {
			continue
		}
		if g := m.GetGauge(); g != nil {
			return g.GetValue(), true
		}
		if c := m.GetCounter(); c != nil {
			return c.GetValue(), true
		}
	}
	return 0, false
}

// plainGauge returns the first gauge value of an unlabeled (or single-series)
// metric family.
func plainGauge(families map[string]*dto.MetricFamily, name string) (float64, bool) {
	family := families[name]
	if family == nil {
		return 0, false
	}
	for _, m := range family.Metric {
		if g := m.GetGauge(); g != nil {
			return g.GetValue(), true
		}
	}
	return 0, false
}

// gpuUtilizationMax returns the maximum GPU utilization across all reported
// GPUs (the hottest GPU is the replica's bottleneck signal).
func gpuUtilizationMax(families map[string]*dto.MetricFamily) (float64, bool) {
	family := families[MetricGPUUtilization]
	if family == nil {
		return 0, false
	}
	found := false
	max := 0.0
	for _, m := range family.Metric {
		if g := m.GetGauge(); g != nil {
			if !found || g.GetValue() > max {
				max = g.GetValue()
			}
			found = true
		}
	}
	if !found {
		return 0, false
	}
	return clamp01(max), true
}

// gpuMemoryPressure returns aggregate used/total GPU memory across all GPUs.
func gpuMemoryPressure(families map[string]*dto.MetricFamily) (float64, bool) {
	used := sumGauge(families, MetricGPUMemoryUsed)
	total := sumGauge(families, MetricGPUMemoryTotal)
	if total <= 0 {
		return 0, false
	}
	return clamp01(used / total), true
}

func sumGauge(families map[string]*dto.MetricFamily, name string) float64 {
	family := families[name]
	if family == nil {
		return 0
	}
	sum := 0.0
	for _, m := range family.Metric {
		if g := m.GetGauge(); g != nil {
			sum += g.GetValue()
		}
	}
	return sum
}

// readCounters extracts the cumulative Triton latency counters for delta math.
func readCounters(families map[string]*dto.MetricFamily) counterSnapshot {
	snap := counterSnapshot{}
	if v, ok := sumCounter(families, MetricRequestDurationUs); ok {
		snap.requestDurationUs = v
		snap.valid = true
	}
	if v, ok := sumCounter(families, MetricQueueDurationUs); ok {
		snap.queueDurationUs = v
		snap.valid = true
	}
	if v, ok := sumCounter(families, MetricRequestSuccess); ok {
		snap.requestCount = v
		snap.valid = true
	}
	return snap
}

func sumCounter(families map[string]*dto.MetricFamily, name string) (float64, bool) {
	family := families[name]
	if family == nil {
		return 0, false
	}
	sum := 0.0
	found := false
	for _, m := range family.Metric {
		if c := m.GetCounter(); c != nil {
			sum += c.GetValue()
			found = true
		}
	}
	return sum, found
}

// counterDeltaAvgSeconds computes (durationDelta / countDelta) in seconds using
// unit divisor applied by the caller via us input (durations are microseconds).
func counterDeltaAvgSeconds(durNow, durPrev, countNow, countPrev float64) (float64, bool) {
	dDur := durNow - durPrev
	dCount := countNow - countPrev
	if dCount <= 0 || dDur < 0 {
		return 0, false
	}
	avgUs := dDur / dCount
	seconds := avgUs / 1e6
	return seconds, true
}

// summarySnapshot extracts quantiles from a Prometheus summary, converting the
// summary's native unit to seconds via unitDivisor (e.g. 1e6 for microseconds).
func summarySnapshot(families map[string]*dto.MetricFamily, name string, unitDivisor float64) (backend.LatencySnapshot, bool) {
	family := families[name]
	if family == nil {
		return backend.LatencySnapshot{}, false
	}
	for _, m := range family.Metric {
		s := m.GetSummary()
		if s == nil {
			continue
		}
		snap := backend.LatencySnapshot{}
		set := false
		for _, q := range s.GetQuantile() {
			v := q.GetValue() / unitDivisor
			if math.IsNaN(v) {
				continue
			}
			val := v
			switch {
			case approx(q.GetQuantile(), 0.5):
				snap.P50Seconds = &val
				set = true
			case approx(q.GetQuantile(), 0.9):
				snap.P90Seconds = &val
				set = true
			case approx(q.GetQuantile(), 0.95):
				snap.P95Seconds = &val
				set = true
			case approx(q.GetQuantile(), 0.99):
				snap.P99Seconds = &val
				set = true
			}
		}
		if set {
			return snap, true
		}
	}
	return backend.LatencySnapshot{}, false
}

// histogramSnapshot computes percentile estimates from a Prometheus histogram,
// converting the histogram's native unit to seconds via unitDivisor (e.g. 1e3
// for milliseconds).
func histogramSnapshot(families map[string]*dto.MetricFamily, name string, unitDivisor float64) (backend.LatencySnapshot, bool) {
	family := families[name]
	if family == nil {
		return backend.LatencySnapshot{}, false
	}
	for _, m := range family.Metric {
		h := m.GetHistogram()
		if h == nil || h.GetSampleCount() == 0 || len(h.Bucket) == 0 {
			continue
		}
		return backend.LatencySnapshot{
			P50Seconds: histogramQuantile(0.50, h, unitDivisor),
			P90Seconds: histogramQuantile(0.90, h, unitDivisor),
			P95Seconds: histogramQuantile(0.95, h, unitDivisor),
			P99Seconds: histogramQuantile(0.99, h, unitDivisor),
		}, true
	}
	return backend.LatencySnapshot{}, false
}

func histogramQuantile(q float64, h *dto.Histogram, unitDivisor float64) *float64 {
	if h == nil || h.GetSampleCount() == 0 || len(h.Bucket) == 0 {
		return nil
	}
	buckets := append([]*dto.Bucket(nil), h.Bucket...)
	sort.Slice(buckets, func(i, j int) bool {
		return buckets[i].GetUpperBound() < buckets[j].GetUpperBound()
	})
	rank := q * float64(h.GetSampleCount())
	for _, b := range buckets {
		if float64(b.GetCumulativeCount()) >= rank {
			v := b.GetUpperBound() / unitDivisor
			return &v
		}
	}
	v := buckets[len(buckets)-1].GetUpperBound() / unitDivisor
	return &v
}

func labelValue(m *dto.Metric, name string) string {
	for _, p := range m.GetLabel() {
		if p.GetName() == name {
			return p.GetValue()
		}
	}
	return ""
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func approx(a, b float64) bool {
	return math.Abs(a-b) < 1e-6
}
