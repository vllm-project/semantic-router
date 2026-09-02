package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"sort"
)

const capacityProfileArtifactName = "capacity-profile.json"

type capacityProfileEvidence struct {
	SchemaVersion string                    `json:"schema_version"`
	Kind          string                    `json:"kind"`
	Protocol      *CapacityLoadProtocol     `json:"protocol"`
	Levels        []capacityProfileLevel    `json:"levels"`
	SLO           *CapacitySLO              `json:"slo"`
	Assessment    capacityProfileAssessment `json:"assessment"`
}

type capacityProfileRepetition struct {
	Concurrency  *int64   `json:"concurrency"`
	Repetition   *int64   `json:"repetition"`
	Requests     *int64   `json:"requests"`
	Successes    *int64   `json:"successes"`
	Errors       *int64   `json:"errors"`
	Elapsed      *float64 `json:"elapsed_seconds"`
	Throughput   *float64 `json:"throughput_rps"`
	LatencyP95MS *float64 `json:"latency_p95_ms"`
	ErrorRate    *float64 `json:"error_rate"`
	ErrorUpper   *float64 `json:"error_rate_upper_bound"`
}

type capacityProfileLevel struct {
	Concurrency               *int64                      `json:"concurrency"`
	WarmupRequests            *int64                      `json:"warmup_requests"`
	WarmupErrors              *int64                      `json:"warmup_errors"`
	WarmupElapsed             *float64                    `json:"warmup_elapsed_seconds"`
	MeasurementRequests       *int64                      `json:"measurement_requests"`
	Successes                 *int64                      `json:"successes"`
	Errors                    *int64                      `json:"errors"`
	Elapsed                   *float64                    `json:"elapsed_seconds"`
	Throughput                *float64                    `json:"throughput_rps"`
	ThroughputCV              *float64                    `json:"throughput_cv"`
	LatencyP50MS              *float64                    `json:"latency_p50_ms"`
	LatencyP95MS              *float64                    `json:"latency_p95_ms"`
	LatencyP99MS              *float64                    `json:"latency_p99_ms"`
	LatencyP95CV              *float64                    `json:"latency_p95_cv"`
	ErrorRate                 *float64                    `json:"error_rate"`
	ErrorRateUpperBound       *float64                    `json:"error_rate_upper_bound"`
	MeasurementClusterCount   *int64                      `json:"measurement_cluster_count"`
	ErrorRateClusterRange     *float64                    `json:"error_rate_cluster_range"`
	InputTokens               *int64                      `json:"input_tokens"`
	OutputTokens              *int64                      `json:"output_tokens"`
	RuntimeCost               *float64                    `json:"runtime_cost_usd"`
	Repetitions               []capacityProfileRepetition `json:"repetitions"`
	ScalingEfficiency         json.RawMessage             `json:"throughput_scaling_efficiency"`
	WarmupPassed              *bool                       `json:"warmup_passed"`
	LatencySLOPassed          *bool                       `json:"latency_slo_passed"`
	ClusterCoveragePassed     *bool                       `json:"cluster_coverage_passed"`
	ErrorRateStabilityPassed  *bool                       `json:"error_rate_stability_passed"`
	ErrorSLOPassed            *bool                       `json:"error_slo_passed"`
	ThroughputSLOPassed       *bool                       `json:"throughput_slo_passed"`
	ScalingSLOPassed          *bool                       `json:"scaling_slo_passed"`
	ThroughputStabilityPassed *bool                       `json:"throughput_stability_passed"`
	LatencyStabilityPassed    *bool                       `json:"latency_stability_passed"`
	Qualified                 *bool                       `json:"qualified"`
}

type capacityBatchKey struct {
	concurrency int64
	phase       string
	repetition  int64
}

type capacityRecordBatch struct {
	rows         int64
	indices      map[int64]struct{}
	successes    int64
	elapsed      *float64
	throughput   *float64
	latencies    []float64
	inputTokens  int64
	outputTokens int64
	runtimeCosts []float64
}

type reducedCapacityLevel struct {
	concurrency         int64
	warmupRequests      int64
	warmupErrors        int64
	warmupElapsed       float64
	measurementRequests int64
	successes           int64
	errors              int64
	elapsed             float64
	throughput          float64
	throughputCV        float64
	latencyP50          float64
	latencyP95          float64
	latencyP99          float64
	latencyP95CV        float64
	errorRate           float64
	errorUpper          float64
	measurementClusters int64
	errorClusterRange   float64
	inputTokens         int64
	outputTokens        int64
	runtimeCost         float64
	repetitions         []capacityProfileRepetition
	scaling             *float64
	warmupPassed        bool
	latencyPassed       bool
	clusterCoverage     bool
	errorRateStable     bool
	errorPassed         bool
	throughputPassed    bool
	scalingPassed       bool
	throughputStable    bool
	latencyStable       bool
	qualified           bool
}

func validateCapacityProfileArtifact(
	runDir string,
	manifest RunManifest,
	report Report,
	records recordAttestation,
) (*capacitySLOAttestation, error) {
	artifact, present := findArtifactByName(report, capacityProfileArtifactName)
	selected := manifest.Mode == ModeLive && containsTrack(manifest.TrackIDs, "capacity")
	if !present {
		if selected {
			return nil, fmt.Errorf("%w: live capacity evidence requires a capacity profile", ErrInvalid)
		}
		return nil, nil
	}
	if !selected || artifact.MediaType != "application/json" {
		return nil, fmt.Errorf("%w: capacity profile does not match a live capacity run", ErrInvalid)
	}
	var profile capacityProfileEvidence
	if err := decodeStrictEvidence(filepath.Join(runDir, capacityProfileArtifactName), &profile); err != nil {
		return nil, fmt.Errorf("%w: invalid capacity profile: %w", ErrInvalid, err)
	}
	if err := validateCapacityProfileContract(profile, manifest); err != nil {
		return nil, fmt.Errorf("%w: invalid capacity profile: %w", ErrInvalid, err)
	}
	batches, err := reduceCapacityRecordBatches(runDir, records)
	if err != nil {
		return nil, fmt.Errorf("%w: capacity profile records are invalid: %w", ErrInvalid, err)
	}
	reduced, err := validateCapacityLevels(profile, batches)
	if err != nil {
		return nil, fmt.Errorf("%w: capacity profile does not match records: %w", ErrInvalid, err)
	}
	headroom, err := validateCapacityAssessment(profile, reduced)
	if err != nil {
		return nil, fmt.Errorf("%w: capacity profile assessment is invalid: %w", ErrInvalid, err)
	}
	attestation := &capacitySLOAttestation{
		Headroom: float64(headroom), LevelCount: len(reduced),
		MinimumClustersPerLevel:  int64(len(reduced[0].repetitions)),
		RequiredClustersPerLevel: profile.Protocol.MinimumMeasurementClustersPerLevel,
		MaxErrorRate:             profile.SLO.MaxErrorRate,
		MaxErrorRateClusterRange: profile.Protocol.MaxErrorRateClusterRange,
	}
	releaseEnvelopeOpen := true
	for _, level := range reduced {
		attestation.MeasurementClusterCount += len(level.repetitions)
		attestation.MinimumClustersPerLevel = min(attestation.MinimumClustersPerLevel, level.measurementClusters)
		attestation.WorstErrorRateUpperBound = math.Max(attestation.WorstErrorRateUpperBound, level.errorUpper)
		attestation.WorstErrorRateClusterRange = math.Max(attestation.WorstErrorRateClusterRange, level.errorClusterRange)
		if releaseEnvelopeOpen {
			attestation.ReleaseErrorRateUpperBound = math.Max(attestation.ReleaseErrorRateUpperBound, level.errorUpper)
			attestation.ReleaseErrorRateClusterRange = math.Max(attestation.ReleaseErrorRateClusterRange, level.errorClusterRange)
			if level.concurrency >= profile.SLO.RequiredConcurrency {
				releaseEnvelopeOpen = false
			}
		}
		for _, repetition := range level.repetitions {
			attestation.MeanErrorRate += *repetition.ErrorRate
		}
	}
	attestation.MeanErrorRate /= float64(attestation.MeasurementClusterCount)
	return attestation, nil
}

func validateCapacityProfileContract(profile capacityProfileEvidence, manifest RunManifest) error {
	if profile.SchemaVersion != SchemaVersion || profile.Kind != "repeated-closed-loop-capacity" ||
		profile.Protocol == nil || profile.SLO == nil || manifest.CapacityLoadProtocol == nil ||
		manifest.CapacitySLO == nil {
		return fmt.Errorf("profile requires the current schema, kind, protocol, and SLO")
	}
	if err := validateCapacityLoadProtocol(profile.Protocol, manifest.Concurrency); err != nil {
		return err
	}
	if err := validateCapacitySLOValue(profile.SLO, manifest.Concurrency); err != nil {
		return err
	}
	if !reflect.DeepEqual(profile.Protocol, manifest.CapacityLoadProtocol) ||
		!reflect.DeepEqual(profile.SLO, manifest.CapacitySLO) {
		return fmt.Errorf("profile protocol or SLO differs from the frozen manifest")
	}
	if len(profile.Levels) != len(profile.Protocol.ConcurrencyLevels) {
		return fmt.Errorf("profile level count differs from the frozen protocol")
	}
	return nil
}

func reduceCapacityRecordBatches(
	runDir string,
	records recordAttestation,
) (map[capacityBatchKey]*capacityRecordBatch, error) {
	if !records.validated {
		return nil, fmt.Errorf("records attestation is unavailable")
	}
	batches := make(map[capacityBatchKey]*capacityRecordBatch)
	rows := 0
	err := scanEvidenceJSONLines(
		filepath.Join(runDir, "records.jsonl"),
		maxWorkerArtifactBytes,
		maxRecordLineBytes,
		maxRecordsPerRun,
		func(line []byte, lineNumber int) error {
			var record executionRecordEvidence
			if err := decodeStrictJSONLine(line, &record); err != nil {
				return fmt.Errorf("line %d: %w", lineNumber, err)
			}
			if record.TrackID != "capacity" {
				return nil
			}
			rows++
			if err := validateCapacityLoadCoordinates(record); err != nil ||
				record.LoadPhase == nil || record.LoadRepetition == nil || record.LoadRequestIndex == nil ||
				record.Concurrency == nil || record.Success == nil || record.LatencyMS == nil ||
				record.ThroughputRPS == nil || record.LoadElapsedSeconds == nil ||
				record.BrokerReceipt == nil || record.EvidenceKind == nil ||
				*record.EvidenceKind != "capacity.closed-loop.v1" {
				return fmt.Errorf("line %d lacks a complete attested capacity load row", lineNumber)
			}
			key := capacityBatchKey{*record.Concurrency, *record.LoadPhase, *record.LoadRepetition}
			batch := batches[key]
			if batch == nil {
				batch = &capacityRecordBatch{indices: make(map[int64]struct{})}
				batches[key] = batch
			}
			if _, duplicate := batch.indices[*record.LoadRequestIndex]; duplicate {
				return fmt.Errorf("line %d duplicates a capacity request index", lineNumber)
			}
			batch.indices[*record.LoadRequestIndex] = struct{}{}
			batch.rows++
			if *record.Success {
				batch.successes++
			}
			if err := bindCapacityBatchFloat(&batch.elapsed, *record.LoadElapsedSeconds); err != nil {
				return fmt.Errorf("line %d elapsed: %w", lineNumber, err)
			}
			if err := bindCapacityBatchFloat(&batch.throughput, *record.ThroughputRPS); err != nil {
				return fmt.Errorf("line %d throughput: %w", lineNumber, err)
			}
			batch.latencies = append(batch.latencies, *record.LatencyMS)
			if record.InputTokens != nil {
				batch.inputTokens += *record.InputTokens
			}
			if record.OutputTokens != nil {
				batch.outputTokens += *record.OutputTokens
			}
			runtimeCost := 0.0
			if record.RuntimeCost != nil {
				runtimeCost = *record.RuntimeCost
			}
			batch.runtimeCosts = append(batch.runtimeCosts, runtimeCost)
			return nil
		},
	)
	if err != nil {
		return nil, err
	}
	if rows != records.ByTrack["capacity"].total() {
		return nil, fmt.Errorf("capacity record count changed after records attestation")
	}
	return batches, nil
}

func bindCapacityBatchFloat(bound **float64, value float64) error {
	if !finiteFloat(value) || value <= 0 {
		return fmt.Errorf("batch value must be finite and positive")
	}
	if *bound == nil {
		copy := value
		*bound = &copy
		return nil
	}
	if !reducedFloatsEqual(**bound, value) {
		return fmt.Errorf("rows disagree on their batch value")
	}
	return nil
}

func validateCapacityLevels(
	profile capacityProfileEvidence,
	batches map[capacityBatchKey]*capacityRecordBatch,
) ([]reducedCapacityLevel, error) {
	reduced := make([]reducedCapacityLevel, 0, len(profile.Levels))
	envelopeOpen := true
	for index, concurrency := range profile.Protocol.ConcurrencyLevels {
		warmupKey := capacityBatchKey{concurrency, "warmup", 0}
		warmup := batches[warmupKey]
		delete(batches, warmupKey)
		expectedWarmup := concurrency * profile.Protocol.WarmupRequestMultiplier
		if err := validateCapacityBatch(warmup, expectedWarmup); err != nil {
			return nil, fmt.Errorf("level %d warmup: %w", index+1, err)
		}
		repetitions := make([]capacityProfileRepetition, 0, profile.Protocol.RepetitionsPerLevel)
		measurement := &capacityRecordBatch{indices: make(map[int64]struct{})}
		throughputs := make([]float64, 0, profile.Protocol.RepetitionsPerLevel)
		p95s := make([]float64, 0, profile.Protocol.RepetitionsPerLevel)
		errorRates := make([]float64, 0, profile.Protocol.RepetitionsPerLevel)
		worstErrorUpper := 0.0
		for repetition := int64(1); repetition <= profile.Protocol.RepetitionsPerLevel; repetition++ {
			key := capacityBatchKey{concurrency, "measurement", repetition}
			batch := batches[key]
			delete(batches, key)
			if err := validateCapacityBatch(batch, profile.Protocol.MeasurementRequestsPerRepetition); err != nil {
				return nil, fmt.Errorf("level %d repetition %d: %w", index+1, repetition, err)
			}
			p95, _ := capacityPercentile(batch.latencies, 0.95)
			errors := batch.rows - batch.successes
			errorRate := float64(errors) / float64(batch.rows)
			errorUpper := capacityOneSidedWilsonUpper(errors, batch.rows)
			repetitionEvidence := capacityProfileRepetition{
				Concurrency: capacityInt64Pointer(concurrency), Repetition: capacityInt64Pointer(repetition),
				Requests: capacityInt64Pointer(batch.rows), Successes: capacityInt64Pointer(batch.successes),
				Errors: capacityInt64Pointer(errors), Elapsed: capacityFloatPointer(*batch.elapsed),
				Throughput: capacityFloatPointer(*batch.throughput), LatencyP95MS: capacityFloatPointer(p95),
				ErrorRate: capacityFloatPointer(errorRate), ErrorUpper: capacityFloatPointer(errorUpper),
			}
			repetitions = append(repetitions, repetitionEvidence)
			throughputs = append(throughputs, *batch.throughput)
			p95s = append(p95s, p95)
			errorRates = append(errorRates, errorRate)
			worstErrorUpper = math.Max(worstErrorUpper, errorUpper)
			mergeCapacityBatch(measurement, batch)
		}
		latencyP50, _ := capacityPercentile(measurement.latencies, 0.50)
		latencyP95, _ := capacityPercentile(measurement.latencies, 0.95)
		latencyP99, _ := capacityPercentile(measurement.latencies, 0.99)
		errors := measurement.rows - measurement.successes
		level := reducedCapacityLevel{
			concurrency: concurrency, warmupRequests: warmup.rows,
			warmupErrors: warmup.rows - warmup.successes, warmupElapsed: *warmup.elapsed,
			measurementRequests: measurement.rows, successes: measurement.successes, errors: errors,
			elapsed: measurementElapsed(repetitions), throughput: capacityMean(throughputs),
			throughputCV: capacitySampleCV(throughputs), latencyP50: latencyP50,
			latencyP95: latencyP95, latencyP99: latencyP99, latencyP95CV: capacitySampleCV(p95s),
			errorRate: capacityMean(errorRates), errorUpper: worstErrorUpper,
			measurementClusters: int64(len(repetitions)),
			errorClusterRange:   capacityRange(errorRates),
			inputTokens:         measurement.inputTokens, outputTokens: measurement.outputTokens,
			runtimeCost: capacityOrderedSum(measurement.runtimeCosts), repetitions: repetitions,
		}
		if index > 0 {
			scaling := (level.throughput / reduced[index-1].throughput) /
				(float64(concurrency) / float64(reduced[index-1].concurrency))
			level.scaling = &scaling
		}
		level.warmupPassed = level.warmupErrors == 0
		level.latencyPassed = level.latencyP95 <= profile.SLO.MaxLatencyP95MS
		level.clusterCoverage = level.measurementClusters >= profile.Protocol.MinimumMeasurementClustersPerLevel
		level.errorRateStable = level.errorClusterRange <= profile.Protocol.MaxErrorRateClusterRange
		level.errorPassed = level.errorUpper <= profile.SLO.MaxErrorRate
		level.throughputPassed = concurrency < profile.SLO.RequiredConcurrency || level.throughput >= profile.SLO.MinThroughputRPS
		level.scalingPassed = level.scaling == nil || *level.scaling >= profile.SLO.MinThroughputScalingEfficiency
		level.throughputStable = level.throughputCV <= profile.Protocol.MaxThroughputCV
		level.latencyStable = level.latencyP95CV <= profile.Protocol.MaxLatencyP95CV
		level.qualified = envelopeOpen && level.warmupPassed && level.latencyPassed &&
			level.clusterCoverage && level.errorRateStable && level.errorPassed &&
			level.throughputPassed && level.scalingPassed &&
			level.throughputStable && level.latencyStable
		if !level.qualified {
			envelopeOpen = false
		}
		if err := compareCapacityLevel(profile.Levels[index], level); err != nil {
			return nil, fmt.Errorf("level %d: %w", index+1, err)
		}
		reduced = append(reduced, level)
	}
	if len(batches) != 0 {
		return nil, fmt.Errorf("records contain an undeclared capacity load batch")
	}
	return reduced, nil
}

func validateCapacityBatch(batch *capacityRecordBatch, expected int64) error {
	if batch == nil || batch.rows != expected || int64(len(batch.indices)) != expected ||
		batch.elapsed == nil || batch.throughput == nil || len(batch.latencies) != int(expected) {
		return fmt.Errorf("batch does not match the frozen request window")
	}
	for index := int64(0); index < expected; index++ {
		if _, present := batch.indices[index]; !present {
			return fmt.Errorf("batch request indices are incomplete")
		}
	}
	if !reducedFloatsEqual(*batch.throughput, float64(expected) / *batch.elapsed) {
		return fmt.Errorf("batch throughput does not match requests and elapsed time")
	}
	return nil
}

func mergeCapacityBatch(target, source *capacityRecordBatch) {
	target.rows += source.rows
	target.successes += source.successes
	target.latencies = append(target.latencies, source.latencies...)
	target.inputTokens += source.inputTokens
	target.outputTokens += source.outputTokens
	target.runtimeCosts = append(target.runtimeCosts, source.runtimeCosts...)
}

func capacityOrderedSum(values []float64) float64 {
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total
}

func measurementElapsed(repetitions []capacityProfileRepetition) float64 {
	total := 0.0
	for _, repetition := range repetitions {
		total += *repetition.Elapsed
	}
	return total
}

func compareCapacityLevel(actual capacityProfileLevel, expected reducedCapacityLevel) error {
	requiredPointers := []any{
		actual.Concurrency, actual.WarmupRequests, actual.WarmupErrors, actual.WarmupElapsed,
		actual.MeasurementRequests, actual.Successes, actual.Errors, actual.Elapsed, actual.Throughput,
		actual.ThroughputCV, actual.LatencyP50MS, actual.LatencyP95MS, actual.LatencyP99MS,
		actual.LatencyP95CV, actual.ErrorRate, actual.ErrorRateUpperBound, actual.InputTokens,
		actual.MeasurementClusterCount, actual.ErrorRateClusterRange,
		actual.OutputTokens, actual.RuntimeCost, actual.WarmupPassed, actual.LatencySLOPassed,
		actual.ClusterCoveragePassed, actual.ErrorRateStabilityPassed,
		actual.ErrorSLOPassed, actual.ThroughputSLOPassed, actual.ScalingSLOPassed,
		actual.ThroughputStabilityPassed, actual.LatencyStabilityPassed, actual.Qualified,
	}
	for _, value := range requiredPointers {
		if reflect.ValueOf(value).IsNil() {
			return fmt.Errorf("required field is missing or null")
		}
	}
	if *actual.Concurrency != expected.concurrency || *actual.WarmupRequests != expected.warmupRequests ||
		*actual.WarmupErrors != expected.warmupErrors || *actual.MeasurementRequests != expected.measurementRequests ||
		*actual.Successes != expected.successes || *actual.Errors != expected.errors ||
		*actual.MeasurementClusterCount != expected.measurementClusters ||
		*actual.InputTokens != expected.inputTokens || *actual.OutputTokens != expected.outputTokens ||
		!reducedFloatsEqual(*actual.WarmupElapsed, expected.warmupElapsed) ||
		!reducedFloatsEqual(*actual.Elapsed, expected.elapsed) ||
		!reducedFloatsEqual(*actual.Throughput, expected.throughput) ||
		!reducedFloatsEqual(*actual.ThroughputCV, expected.throughputCV) ||
		!reducedFloatsEqual(*actual.LatencyP50MS, expected.latencyP50) ||
		!reducedFloatsEqual(*actual.LatencyP95MS, expected.latencyP95) ||
		!reducedFloatsEqual(*actual.LatencyP99MS, expected.latencyP99) ||
		!reducedFloatsEqual(*actual.LatencyP95CV, expected.latencyP95CV) ||
		!reducedFloatsEqual(*actual.ErrorRate, expected.errorRate) ||
		!reducedFloatsEqual(*actual.ErrorRateUpperBound, expected.errorUpper) ||
		!reducedFloatsEqual(*actual.ErrorRateClusterRange, expected.errorClusterRange) ||
		!reducedFloatsEqual(*actual.RuntimeCost, expected.runtimeCost) {
		return fmt.Errorf("profile measurements differ from request records")
	}
	if err := compareCapacityRepetitions(actual.Repetitions, expected.repetitions); err != nil {
		return err
	}
	actualScaling, present, err := decodeCapacityOptionalFloat("throughput_scaling_efficiency", actual.ScalingEfficiency)
	if err != nil || present != (expected.scaling != nil) ||
		(present && !reducedFloatsEqual(actualScaling, *expected.scaling)) {
		return fmt.Errorf("throughput scaling differs from request records")
	}
	actualFlags := []bool{
		*actual.WarmupPassed, *actual.LatencySLOPassed, *actual.ClusterCoveragePassed,
		*actual.ErrorRateStabilityPassed, *actual.ErrorSLOPassed,
		*actual.ThroughputSLOPassed, *actual.ScalingSLOPassed,
		*actual.ThroughputStabilityPassed, *actual.LatencyStabilityPassed, *actual.Qualified,
	}
	expectedFlags := []bool{
		expected.warmupPassed, expected.latencyPassed, expected.clusterCoverage,
		expected.errorRateStable, expected.errorPassed,
		expected.throughputPassed, expected.scalingPassed,
		expected.throughputStable, expected.latencyStable, expected.qualified,
	}
	if !reflect.DeepEqual(actualFlags, expectedFlags) {
		return fmt.Errorf("profile decisions differ from server reduction")
	}
	return nil
}

func compareCapacityRepetitions(actual, expected []capacityProfileRepetition) error {
	if len(actual) != len(expected) {
		return fmt.Errorf("repetition count differs from the frozen protocol")
	}
	for index := range actual {
		left, right := actual[index], expected[index]
		if left.Concurrency == nil || left.Repetition == nil || left.Requests == nil ||
			left.Successes == nil || left.Errors == nil || left.Elapsed == nil ||
			left.Throughput == nil || left.LatencyP95MS == nil || left.ErrorRate == nil ||
			left.ErrorUpper == nil ||
			*left.Concurrency != *right.Concurrency || *left.Repetition != *right.Repetition ||
			*left.Requests != *right.Requests || *left.Successes != *right.Successes ||
			*left.Errors != *right.Errors || !reducedFloatsEqual(*left.Elapsed, *right.Elapsed) ||
			!reducedFloatsEqual(*left.Throughput, *right.Throughput) ||
			!reducedFloatsEqual(*left.LatencyP95MS, *right.LatencyP95MS) ||
			!reducedFloatsEqual(*left.ErrorRate, *right.ErrorRate) ||
			!reducedFloatsEqual(*left.ErrorUpper, *right.ErrorUpper) {
			return fmt.Errorf("repetition %d differs from request records", index+1)
		}
	}
	return nil
}

func capacityMean(values []float64) float64 {
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}

func capacityRange(values []float64) float64 {
	minimum, maximum := values[0], values[0]
	for _, value := range values[1:] {
		minimum = math.Min(minimum, value)
		maximum = math.Max(maximum, value)
	}
	return maximum - minimum
}

func capacityInt64Pointer(value int64) *int64 { return &value }

func capacityFloatPointer(value float64) *float64 { return &value }

func capacitySampleCV(values []float64) float64 {
	mean := capacityMean(values)
	if mean == 0 {
		return 0
	}
	variance := 0.0
	for _, value := range values {
		variance += (value - mean) * (value - mean)
	}
	variance /= float64(len(values) - 1)
	return math.Sqrt(math.Max(variance, 0)) / mean
}

func capacityOneSidedWilsonUpper(events, total int64) float64 {
	const z = 1.6448536269514722
	n := float64(total)
	estimate := float64(events) / n
	z2 := z * z
	denominator := 1 + z2/n
	center := estimate + z2/(2*n)
	margin := z * math.Sqrt(estimate*(1-estimate)/n+z2/(4*n*n))
	return math.Min(1, (center+margin)/denominator)
}

func capacityPercentile(values []float64, quantile float64) (float64, bool) {
	if len(values) == 0 {
		return 0, false
	}
	ordered := append([]float64(nil), values...)
	sort.Float64s(ordered)
	if len(ordered) == 1 {
		return ordered[0], true
	}
	position := float64(len(ordered)-1) * quantile
	lower := int(position)
	upper := lower + 1
	if upper >= len(ordered) {
		upper = len(ordered) - 1
	}
	fraction := position - float64(lower)
	return ordered[lower] + (ordered[upper]-ordered[lower])*fraction, true
}

func decodeCapacityOptionalFloat(name string, raw json.RawMessage) (float64, bool, error) {
	if len(raw) == 0 {
		return 0, false, fmt.Errorf("%s is required", name)
	}
	trimmed := bytes.TrimSpace(raw)
	if bytes.Equal(trimmed, []byte("null")) {
		return 0, false, nil
	}
	var value float64
	decoder := json.NewDecoder(bytes.NewReader(trimmed))
	if err := decoder.Decode(&value); err != nil {
		return 0, false, fmt.Errorf("%s must be numeric or null", name)
	}
	if err := ensureJSONEOF(decoder); err != nil || !finiteFloat(value) || value < 0 {
		return 0, false, fmt.Errorf("%s must be finite, non-negative, or null", name)
	}
	return value, true, nil
}
