package evaluationplane

import (
	"fmt"
	"math"
	"sort"
)

type recoveryMethodAttestation struct {
	PairCount                 int
	ClusterCount              int
	LedgerTotalPairCount      int
	DistinctSeedCount         int
	MinimumClusterCount       int
	MinimumDistinctSeedCount  int
	MinimumPairCount          int
	ClusterPassRate           *float64
	ClusterPassRateLower95    *float64
	ClusterPassRateUpper95    *float64
	TreatmentSuccessRate      *float64
	BaselineSuccessRate       *float64
	SuccessDelta              *float64
	MeanLatencyDeltaMS        *float64
	MaximumRetryObserved      *float64
	MinimumPassRateLower95    float64
	MaximumRecoveryLatencyMS  *float64
	MaximumRetryAmplification *float64
	PolicySnapshotDigest      string
	ConfigDigest              string
	TargetID                  string
	BackendTopologyDigest     string
	MixtureSnapshotDigest     string
	Passed                    *bool
}

func recoveryContractEqual(left, right recoveryMethodEvidence) bool {
	return left.LedgerID == right.LedgerID && left.SourceID == right.SourceID &&
		left.PolicySnapshotDigest == right.PolicySnapshotDigest && left.ConfigDigest == right.ConfigDigest &&
		left.TargetID == right.TargetID && left.BackendTopologyDigest == right.BackendTopologyDigest &&
		left.MixtureSnapshotDigest == right.MixtureSnapshotDigest &&
		left.LedgerTotalPairCount == right.LedgerTotalPairCount && left.MinimumPairCount == right.MinimumPairCount &&
		left.MinimumClusterCount == right.MinimumClusterCount &&
		left.MinimumDistinctSeedCount == right.MinimumDistinctSeedCount &&
		reducedFloatsEqual(left.MaximumRecoveryLatencyMS, right.MaximumRecoveryLatencyMS) &&
		reducedFloatsEqual(left.MaximumRetryAmplification, right.MaximumRetryAmplification)
}

func oneSidedWilsonBounds(successes, total int) (*float64, *float64) {
	if total <= 0 {
		return nil, nil
	}
	const z = 1.6448536269514722
	proportion := float64(successes) / float64(total)
	zSquared := z * z
	denominator := 1 + zSquared/float64(total)
	center := proportion + zSquared/(2*float64(total))
	margin := z * math.Sqrt(proportion*(1-proportion)/float64(total)+zSquared/(4*float64(total*total)))
	lower := math.Max(0, (center-margin)/denominator)
	upper := math.Min(1, (center+margin)/denominator)
	return &lower, &upper
}

func oneSidedWilsonLower(successes, total int) *float64 {
	lower, _ := oneSidedWilsonBounds(successes, total)
	return lower
}

type recoveryClusterObservation struct {
	Passed                bool
	BaselineSucceeded     bool
	TreatmentSucceeded    bool
	MaximumLatencyDeltaMS float64
}

func summarizeRecoveryClusters(clusters map[string]recoveryClusterObservation) (int, int, int, float64) {
	clusterIDs := make([]string, 0, len(clusters))
	for clusterID := range clusters {
		clusterIDs = append(clusterIDs, clusterID)
	}
	sort.Strings(clusterIDs)
	clusterPasses := 0
	baselineSuccesses := 0
	treatmentSuccesses := 0
	latencyDeltaTotal := 0.0
	for _, clusterID := range clusterIDs {
		cluster := clusters[clusterID]
		if cluster.Passed {
			clusterPasses++
		}
		if cluster.BaselineSucceeded {
			baselineSuccesses++
		}
		if cluster.TreatmentSucceeded {
			treatmentSuccesses++
		}
		latencyDeltaTotal += cluster.MaximumLatencyDeltaMS
	}
	return clusterPasses, baselineSuccesses, treatmentSuccesses, latencyDeltaTotal
}

// reduceRecoveryMethod treats cluster_id, not a receipt pair, as the
// independent analysis unit. A cluster passes only when every sealed pair
// passes; its latency contribution is the worst pair delta. Sorting cluster
// identities before summation keeps the server and worker reductions stable.
func reduceRecoveryMethod(records []executionRecordEvidence) (recoveryMethodAttestation, error) {
	var first *recoveryMethodEvidence
	faultIDs := make(map[string]struct{})
	pairIDs := make(map[string]struct{})
	seeds := make(map[int64]struct{})
	clusters := make(map[string]recoveryClusterObservation)
	maximumRetryObserved := 0.0
	count := 0
	for _, record := range records {
		method := record.Recovery
		if record.TrackID != "agentic" || method == nil {
			continue
		}
		if first == nil {
			copyMethod := *method
			first = &copyMethod
		} else if !recoveryContractEqual(*first, *method) {
			return recoveryMethodAttestation{}, fmt.Errorf("recovery rows mix sealed ledger contracts")
		}
		pairKey := method.CohortPairID + "\x00" + method.RepetitionID
		if _, duplicate := faultIDs[method.FaultID]; duplicate {
			return recoveryMethodAttestation{}, fmt.Errorf("recovery fault identities must be unique")
		}
		if _, duplicate := pairIDs[pairKey]; duplicate {
			return recoveryMethodAttestation{}, fmt.Errorf("recovery cohort/repetition pairs must be unique")
		}
		faultIDs[method.FaultID] = struct{}{}
		pairIDs[pairKey] = struct{}{}
		seeds[method.Seed] = struct{}{}
		retryAmplification := float64(method.TreatmentRetryCount+1) / float64(method.BaselineRetryCount+1)
		passed := method.InjectionObserved && method.Recovered && method.StatePreserved && method.TreatmentTerminalSuccess &&
			method.DuplicateSideEffectCount == 0 && method.TreatmentRecoveryLatencyMS <= method.MaximumRecoveryLatencyMS &&
			retryAmplification <= method.MaximumRetryAmplification
		if record.Success == nil || *record.Success != passed {
			return recoveryMethodAttestation{}, fmt.Errorf("agentic result does not bind its recovery evidence")
		}
		latencyDelta := method.TreatmentRecoveryLatencyMS - method.BaselineRecoveryLatencyMS
		cluster, present := clusters[method.ClusterID]
		if !present {
			cluster = recoveryClusterObservation{
				Passed: passed, BaselineSucceeded: method.BaselineTerminalSuccess,
				TreatmentSucceeded: method.TreatmentTerminalSuccess, MaximumLatencyDeltaMS: latencyDelta,
			}
		} else {
			cluster.Passed = cluster.Passed && passed
			cluster.BaselineSucceeded = cluster.BaselineSucceeded && method.BaselineTerminalSuccess
			cluster.TreatmentSucceeded = cluster.TreatmentSucceeded && method.TreatmentTerminalSuccess
			if latencyDelta > cluster.MaximumLatencyDeltaMS {
				cluster.MaximumLatencyDeltaMS = latencyDelta
			}
		}
		clusters[method.ClusterID] = cluster
		if retryAmplification > maximumRetryObserved {
			maximumRetryObserved = retryAmplification
		}
		count++
	}
	if first == nil {
		return recoveryMethodAttestation{}, nil
	}
	clusterPasses, clusterBaselineSuccesses, clusterTreatmentSuccesses, latencyDeltaTotal := summarizeRecoveryClusters(clusters)
	clusterCount := len(clusters)
	clusterPassRate := float64(clusterPasses) / float64(clusterCount)
	baselineRate := float64(clusterBaselineSuccesses) / float64(clusterCount)
	treatmentRate := float64(clusterTreatmentSuccesses) / float64(clusterCount)
	successDelta := treatmentRate - baselineRate
	meanLatencyDelta := latencyDeltaTotal / float64(clusterCount)
	lower, upper := oneSidedWilsonBounds(clusterPasses, clusterCount)
	maxLatency := first.MaximumRecoveryLatencyMS
	maxRetry := first.MaximumRetryAmplification
	attestation := recoveryMethodAttestation{
		PairCount: count, ClusterCount: clusterCount, LedgerTotalPairCount: first.LedgerTotalPairCount,
		DistinctSeedCount: len(seeds), MinimumDistinctSeedCount: first.MinimumDistinctSeedCount,
		MinimumClusterCount: first.MinimumClusterCount, MinimumPairCount: first.MinimumPairCount,
		ClusterPassRate: &clusterPassRate, ClusterPassRateLower95: lower, ClusterPassRateUpper95: upper,
		TreatmentSuccessRate: &treatmentRate, BaselineSuccessRate: &baselineRate,
		SuccessDelta: &successDelta, MeanLatencyDeltaMS: &meanLatencyDelta,
		MaximumRetryObserved:     &maximumRetryObserved,
		MinimumPassRateLower95:   minimumRecoveryPassRateLowerBound,
		MaximumRecoveryLatencyMS: &maxLatency, MaximumRetryAmplification: &maxRetry,
		PolicySnapshotDigest: first.PolicySnapshotDigest, ConfigDigest: first.ConfigDigest,
		TargetID: first.TargetID, BackendTopologyDigest: first.BackendTopologyDigest,
		MixtureSnapshotDigest: first.MixtureSnapshotDigest,
	}
	if count == first.LedgerTotalPairCount && count >= first.MinimumPairCount && clusterCount >= first.MinimumClusterCount &&
		len(seeds) >= first.MinimumDistinctSeedCount && lower != nil {
		passed := *lower >= minimumRecoveryPassRateLowerBound
		attestation.Passed = &passed
	}
	return attestation, nil
}
