package evaluationplane

import (
	"fmt"
	"math"
	"reflect"
)

const (
	capacityLoadKind                   = "closed-loop"
	capacityLoadConfidence             = 0.95
	minimumCapacityWarmupMultiplier    = int64(2)
	maximumCapacityWarmupMultiplier    = int64(4)
	minimumCapacityMeasurementRequests = int64(100)
	maximumCapacityMeasurementRequests = int64(500)
	minimumCapacityRepetitions         = int64(3)
	maximumCapacityRepetitions         = int64(5)
	minimumCapacityMeasurementClusters = int64(3)
	capacityMaxErrorRateClusterRange   = 0.05
	maximumCapacityStabilityCV         = 0.20
)

func validateCapacityRunContract(
	mode Mode,
	trackIDs []TrackID,
	concurrency int,
	slo *CapacitySLO,
	protocol *CapacityLoadProtocol,
) error {
	selected := mode == ModeLive && containsTrack(trackIDs, "capacity")
	if !selected {
		if slo != nil || protocol != nil {
			return fmt.Errorf(
				"%w: capacity_slo and capacity_load_protocol are valid only for a live capacity track",
				ErrInvalid,
			)
		}
		return nil
	}
	if concurrency < 2 {
		return fmt.Errorf("%w: live capacity track requires concurrency of at least 2", ErrInvalid)
	}
	if err := validateCapacitySLOValue(slo, concurrency); err != nil {
		return err
	}
	return validateCapacityLoadProtocol(protocol, concurrency)
}

func validateCapacitySLOValue(slo *CapacitySLO, concurrency int) error {
	if slo == nil {
		return fmt.Errorf("%w: live capacity track requires capacity_slo", ErrInvalid)
	}
	if slo.SchemaVersion != SchemaVersion || slo.RequiredConcurrency < 1 ||
		slo.RequiredConcurrency > int64(concurrency) ||
		!positiveFinite(slo.MaxLatencyP95MS) ||
		!finiteFloat(slo.MaxErrorRate) || slo.MaxErrorRate < 0 || slo.MaxErrorRate >= 1 ||
		!positiveFinite(slo.MinThroughputRPS) ||
		!positiveFinite(slo.MinThroughputScalingEfficiency) ||
		slo.MinThroughputScalingEfficiency > 1 {
		return fmt.Errorf("%w: capacity_slo contains invalid operating bounds", ErrInvalid)
	}
	return nil
}

func validateCapacityLoadProtocol(protocol *CapacityLoadProtocol, concurrency int) error {
	if protocol == nil {
		return fmt.Errorf("%w: live capacity track requires capacity_load_protocol", ErrInvalid)
	}
	expectedLevels, err := capacityConcurrencyLevels(concurrency)
	if err != nil {
		return fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	if protocol.SchemaVersion != SchemaVersion || protocol.Kind != capacityLoadKind ||
		!reflect.DeepEqual(protocol.ConcurrencyLevels, expectedLevels) ||
		protocol.WarmupRequestMultiplier < minimumCapacityWarmupMultiplier ||
		protocol.WarmupRequestMultiplier > maximumCapacityWarmupMultiplier ||
		protocol.MeasurementRequestsPerRepetition < minimumCapacityMeasurementRequests ||
		protocol.MeasurementRequestsPerRepetition > maximumCapacityMeasurementRequests ||
		protocol.RepetitionsPerLevel < minimumCapacityRepetitions ||
		protocol.RepetitionsPerLevel > maximumCapacityRepetitions ||
		protocol.MinimumMeasurementClustersPerLevel != minimumCapacityMeasurementClusters ||
		protocol.MinimumMeasurementClustersPerLevel > protocol.RepetitionsPerLevel ||
		protocol.ConfidenceLevel != capacityLoadConfidence ||
		protocol.MaxErrorRateClusterRange != capacityMaxErrorRateClusterRange ||
		!positiveFinite(protocol.MaxThroughputCV) ||
		protocol.MaxThroughputCV > maximumCapacityStabilityCV ||
		!positiveFinite(protocol.MaxLatencyP95CV) ||
		protocol.MaxLatencyP95CV > maximumCapacityStabilityCV {
		return fmt.Errorf("%w: capacity_load_protocol violates the platform measurement contract", ErrInvalid)
	}
	if capacityLoadRequestBudget(*protocol) > maxWorkerBrokerRequests {
		return fmt.Errorf("%w: capacity_load_protocol exceeds the broker request budget", ErrInvalid)
	}
	return nil
}

func capacityConcurrencyLevels(maximum int) ([]int64, error) {
	if maximum < 2 || maximum > maxRunConcurrency {
		return nil, fmt.Errorf("capacity maximum concurrency must be between 2 and %d", maxRunConcurrency)
	}
	levels := []int64{1}
	for level := 2; level < maximum; level *= 2 {
		levels = append(levels, int64(level))
	}
	levels = append(levels, int64(maximum))
	return levels, nil
}

func capacityLoadRequestBudget(protocol CapacityLoadProtocol) int64 {
	warmup := int64(0)
	for _, level := range protocol.ConcurrencyLevels {
		warmup += level * protocol.WarmupRequestMultiplier
	}
	measured := int64(len(protocol.ConcurrencyLevels)) *
		protocol.MeasurementRequestsPerRepetition * protocol.RepetitionsPerLevel
	return warmup + measured
}

func positiveFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value > 0
}

func copyCapacitySLO(source *CapacitySLO) *CapacitySLO {
	if source == nil {
		return nil
	}
	clone := *source
	return &clone
}

func copyCapacityLoadProtocol(source *CapacityLoadProtocol) *CapacityLoadProtocol {
	if source == nil {
		return nil
	}
	clone := *source
	clone.ConcurrencyLevels = append([]int64(nil), source.ConcurrencyLevels...)
	return &clone
}
