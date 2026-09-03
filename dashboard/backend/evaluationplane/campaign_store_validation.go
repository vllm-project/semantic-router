package evaluationplane

import (
	"fmt"
	"reflect"
	"strings"
)

func validateCampaignDecisionContract(campaign Campaign) error {
	request := CreateCampaignRequest{
		ClientRequestID: campaign.ID, Name: campaign.Name, Description: campaign.Description,
		ChangeProfile: campaign.ChangeProfile, GateBindings: campaign.GateBindings,
	}
	if err := validateCampaignRequest(request); err != nil {
		return err
	}
	decision := campaign.Decision
	if !decision.CreatedAt.Equal(campaign.CreatedAt) || decision.Summary == "" ||
		!validDecisionVerdict(decision.Verdict) {
		return fmt.Errorf("%w: evaluation campaign decision metadata or verdict is invalid", ErrInvalid)
	}
	for index, gate := range decision.Gates {
		if err := validateStoredCampaignGate(campaign.ChangeProfile, index, gate); err != nil {
			return err
		}
	}
	verdict, summary, recommendations := campaignDecisionSummary(decision.Gates)
	if decision.Verdict != verdict || decision.Summary != summary ||
		!reflect.DeepEqual(decision.Recommendations, recommendations) {
		return fmt.Errorf("%w: evaluation campaign decision summary contradicts its gates", ErrInvalid)
	}
	anchors, err := validateCampaignEvidenceAnchors(campaign.GateBindings, decision.Evidence)
	if err != nil {
		return err
	}
	if err := validateCampaignGateSourceOwnership(campaign, anchors); err != nil {
		return err
	}
	expectsPaired := campaign.GateBindings.G3ControlledPair != nil
	if expectsPaired != (decision.PairedLiveEvidence != nil) {
		return fmt.Errorf("%w: evaluation campaign paired-live receipt presence is invalid", ErrInvalid)
	}
	if decision.PairedLiveEvidence != nil {
		if err := validateCampaignPairedLiveEvidence(campaign, anchors, *decision.PairedLiveEvidence); err != nil {
			return err
		}
		if err := validateCampaignPairedGateBinding(campaign, anchors, *decision.PairedLiveEvidence); err != nil {
			return err
		}
	}
	expectsFidelity := campaign.GateBindings.G5Fidelity != nil
	if expectsFidelity != (decision.FidelityEvidence != nil) {
		return fmt.Errorf("%w: evaluation campaign fidelity receipt presence is invalid", ErrInvalid)
	}
	if decision.FidelityEvidence != nil {
		if err := validateCampaignFidelityEvidence(campaign, anchors, *decision.FidelityEvidence); err != nil {
			return err
		}
	}
	return nil
}

func validateStoredCampaignGate(profile ChangeProfile, index int, gate CampaignGate) error {
	if index >= len(requiredGateIDs) || gate.ID != requiredGateIDs[index] ||
		gate.Source == "" || gate.Source != strings.TrimSpace(gate.Source) || gate.EvidenceRefs == nil ||
		gate.Rationale == "" || gate.Rationale != strings.TrimSpace(gate.Rationale) || gate.SampleCount < 0 {
		return fmt.Errorf("%w: evaluation campaign gate contract is invalid", ErrInvalid)
	}
	definition, defined := releaseGateDefinitionByID(gate.ID)
	disposition, profiled := releaseProfileDisposition(profile, gate.ID)
	if !defined || !profiled {
		return fmt.Errorf("%w: evaluation campaign gate definition is missing", ErrInvalid)
	}
	expectedName, minimum := definition.Name, EvidenceLevel("E5")
	if gate.ID != "G0" && gate.ID != "G1" {
		slot, ok := campaignSlotContract(profile, gate.ID)
		if !ok {
			return fmt.Errorf("%w: evaluation campaign slot %s is missing", ErrInvalid, gate.ID)
		}
		expectedName, disposition, minimum = slot.Name, slot.Disposition, slot.MinimumEvidenceLevel
	}
	if gate.Name != expectedName || gate.Disposition != disposition ||
		evidenceLevelRank(gate.EvidenceLevel) < evidenceLevelRank(minimum) {
		return fmt.Errorf("%w: evaluation campaign gate %s metadata is invalid", ErrInvalid, gate.ID)
	}
	if gate.Disposition == GateDispositionNotApplicable {
		if gate.Verdict != GateVerdictNotApplicable || gate.Observed != nil || gate.Threshold != nil || len(gate.EvidenceRefs) != 0 {
			return fmt.Errorf("%w: not-applicable campaign gate claims a decision", ErrInvalid)
		}
	} else if !validGateVerdict(gate.Verdict) || gate.Verdict == GateVerdictNotApplicable {
		return fmt.Errorf("%w: evaluation campaign gate verdict is invalid", ErrInvalid)
	}
	if (gate.Observed == nil) != (gate.Threshold == nil) {
		return fmt.Errorf("%w: evaluation campaign gate observation is incomplete", ErrInvalid)
	}
	if gate.Observed != nil && (!finiteFloat(*gate.Observed) || !finiteFloat(gate.Threshold.Value) ||
		(gate.Threshold.Operator != ">=" && gate.Threshold.Operator != "<=") || gate.Threshold.Unit == "") {
		return fmt.Errorf("%w: evaluation campaign gate threshold is invalid", ErrInvalid)
	}
	for _, reference := range gate.EvidenceRefs {
		if reference == "" || reference != strings.TrimSpace(reference) {
			return fmt.Errorf("%w: evaluation campaign gate evidence reference is invalid", ErrInvalid)
		}
	}
	return nil
}

func validateCampaignEvidenceAnchors(
	gateBindings CampaignGateBindings,
	evidence []CampaignEvidenceAnchor,
) (map[string]CampaignEvidenceAnchor, error) {
	expected, err := campaignEvidenceBindings(gateBindings)
	if err != nil {
		return nil, err
	}
	if len(evidence) != len(expected) {
		return nil, fmt.Errorf("%w: evaluation campaign evidence slots are incomplete", ErrInvalid)
	}
	anchors := make(map[string]CampaignEvidenceAnchor, len(evidence))
	candidateSubject := ""
	for index, anchor := range evidence {
		binding := expected[index]
		key := campaignEvidenceKey(binding.slotID, binding.bindingRole)
		requiresLive := binding.gateID == "G3" || binding.gateID == "G2" ||
			binding.gateID == "G6" || binding.gateID == "G7" || binding.gateID == "G8" ||
			binding.gateID == "G9" || (binding.gateID == "G5" && binding.bindingRole == "live")
		if anchor.SlotID != binding.slotID || anchor.GateID != binding.gateID ||
			anchor.BindingRole != binding.bindingRole || anchor.RunID != binding.runID ||
			!digestPattern.MatchString(anchor.ManifestSemanticDigest) ||
			!digestPattern.MatchString(anchor.ManifestArtifactDigest) ||
			!digestPattern.MatchString(anchor.ReportDigest) ||
			!digestPattern.MatchString(anchor.PrivateReceiptDigest) ||
			(binding.candidate && !digestPattern.MatchString(anchor.CandidateSubjectDigest)) ||
			(!binding.candidate && anchor.CandidateSubjectDigest != "") ||
			(requiresLive && !digestPattern.MatchString(anchor.ExecutionAttestationDigest)) ||
			(anchor.ExecutionAttestationDigest != "" && !digestPattern.MatchString(anchor.ExecutionAttestationDigest)) {
			return nil, fmt.Errorf("%w: evaluation campaign evidence anchor %s is invalid", ErrInvalid, key)
		}
		if binding.candidate {
			if candidateSubject == "" {
				candidateSubject = anchor.CandidateSubjectDigest
			} else if anchor.CandidateSubjectDigest != candidateSubject {
				return nil, fmt.Errorf("%w: evaluation campaign anchors do not identify one exact subject", ErrInvalid)
			}
		}
		anchors[key] = anchor
	}
	return anchors, nil
}

func validateCampaignGateSourceOwnership(
	campaign Campaign,
	anchors map[string]CampaignEvidenceAnchor,
) error {
	for _, gate := range campaign.Decision.Gates {
		if gate.Disposition == GateDispositionNotApplicable {
			if gate.Source != "campaign_contract" {
				return fmt.Errorf("%w: campaign gate %s has invalid not-applicable ownership", ErrInvalid, gate.ID)
			}
			continue
		}
		expectedSource := ""
		expectedRefs := []string{}
		slotID := "g" + gate.ID[1:]
		switch gate.ID {
		case "G0", "G1":
			expectedSource = "server_anchors"
			expectedRefs = campaignAnchorRefs(campaign.GateBindings, anchors)
		case "G3":
			if campaign.GateBindings.G3ControlledPair == nil {
				expectedSource = "campaign_slot"
			} else {
				expectedSource = "server_attested_paired_live"
				expectedRefs = append(
					campaignAnchorRefsForKeys(anchors, "g3:baseline", "g3:candidate"),
					"campaign-paired-live:"+campaign.Decision.PairedLiveEvidence.Digest,
				)
			}
		case "G5":
			if campaign.GateBindings.G5Fidelity == nil {
				expectedSource = "campaign_slot"
			} else {
				expectedSource = "reference_to_fresh_live"
				expectedRefs = append(
					campaignAnchorRefsForKeys(anchors, "g5:reference", "g5:live"),
					"campaign-fidelity:"+campaign.Decision.FidelityEvidence.Digest,
				)
			}
		default:
			if campaignSlotBound(campaign.GateBindings, gate.ID) {
				expectedSource = "campaign_slot:" + slotID
				expectedRefs = campaignAnchorRefsForKeys(anchors, slotID+":"+campaignSingleBindingRole)
			} else {
				expectedSource = "campaign_slot"
			}
		}
		if gate.Source != expectedSource || !reflect.DeepEqual(gate.EvidenceRefs, expectedRefs) {
			return fmt.Errorf("%w: campaign gate %s violates source ownership", ErrInvalid, gate.ID)
		}
		if !campaignSlotBound(campaign.GateBindings, gate.ID) && gate.ID != "G0" && gate.ID != "G1" &&
			(gate.Verdict != "unavailable" || gate.Observed != nil || gate.Threshold != nil || gate.SampleCount != 0) {
			return fmt.Errorf("%w: campaign gate %s overclaims an unbound advisory slot", ErrInvalid, gate.ID)
		}
	}
	return nil
}

func campaignAnchorRefs(
	gateBindings CampaignGateBindings,
	anchors map[string]CampaignEvidenceAnchor,
) []string {
	bindings, _ := campaignEvidenceBindings(gateBindings)
	keys := make([]string, 0, len(bindings))
	for _, binding := range bindings {
		keys = append(keys, campaignEvidenceKey(binding.slotID, binding.bindingRole))
	}
	return campaignAnchorRefsForKeys(anchors, keys...)
}

func campaignAnchorRefsForKeys(anchors map[string]CampaignEvidenceAnchor, keys ...string) []string {
	refs := []string{}
	for _, key := range keys {
		if anchor, found := anchors[key]; found {
			refs = append(refs, anchorEvidenceRefs(anchor)...)
		}
	}
	return refs
}

func validateCampaignPairedLiveEvidence(
	campaign Campaign,
	anchors map[string]CampaignEvidenceAnchor,
	evidence CampaignPairedLiveEvidence,
) error {
	pair := campaign.GateBindings.G3ControlledPair
	if pair == nil {
		return fmt.Errorf("%w: paired-live evidence has no G3 binding", ErrInvalid)
	}
	baseline, candidate := anchors["g3:baseline"], anchors["g3:candidate"]
	if evidence.SchemaVersion != SchemaVersion || evidence.ContractVersion != CampaignPairedLiveContractVersion ||
		!validClientRequestID(evidence.ControlledPairSessionID) ||
		evidence.ControlledPairProtocol != controlledPairInterleaveABBA ||
		evidence.BaselineRunID != pair.BaselineRunID || evidence.CandidateRunID != pair.CandidateRunID ||
		evidence.CandidateSubjectDigest != candidate.CandidateSubjectDigest ||
		!evidenceIDPattern.MatchString(evidence.BaselineTargetID) ||
		!evidenceIDPattern.MatchString(evidence.CandidateTargetID) ||
		evidence.BaselineTargetID == evidence.CandidateTargetID ||
		!evidenceIDPattern.MatchString(evidence.MixtureID) ||
		evidence.RecipeName == "" || evidence.RecipeName != strings.TrimSpace(evidence.RecipeName) ||
		len(evidence.RecipeName) > 512 ||
		!validStoredTrackIDs(evidence.TrackIDs) ||
		evidence.WorkloadSnapshotDigest == "" || !digestPattern.MatchString(evidence.WorkloadSnapshotDigest) ||
		len(evidence.BenchmarkRevisions) == 0 || evidence.BootstrapSamples != campaignPairedBootstrapSamples ||
		evidence.ConfidenceLevel != campaignPairedConfidenceLevel ||
		evidence.PromotionPolicy != frozenCampaignG3PromotionPolicy ||
		evidence.BaselineManifestDigest != baseline.ManifestSemanticDigest ||
		evidence.CandidateManifestDigest != candidate.ManifestSemanticDigest ||
		evidence.BaselineExecutionAttestationDigest != baseline.ExecutionAttestationDigest ||
		evidence.CandidateExecutionAttestationDigest != candidate.ExecutionAttestationDigest ||
		!campaignPairedProvenanceDigestsValid(evidence) || evidence.Statistics == nil ||
		evidence.PromotionStatistics == nil || evidence.ModelPoolArmReliability == nil ||
		evidence.BaselineCodeRevision == "" || evidence.BaselineCodeRevision != strings.TrimSpace(evidence.BaselineCodeRevision) ||
		evidence.CandidateCodeRevision == "" || evidence.CandidateCodeRevision != strings.TrimSpace(evidence.CandidateCodeRevision) ||
		!digestPattern.MatchString(evidence.Digest) {
		return fmt.Errorf("%w: evaluation campaign paired-live provenance is invalid", ErrInvalid)
	}
	for suiteID, revision := range evidence.BenchmarkRevisions {
		if !portableSuiteIDPattern.MatchString(suiteID) || revision == "" || revision != strings.TrimSpace(revision) {
			return fmt.Errorf("%w: evaluation campaign benchmark revision is invalid", ErrInvalid)
		}
	}
	for _, trackID := range evidence.TrackIDs {
		if !campaignTrackHasExecutionContract(trackID) {
			return fmt.Errorf("%w: paired-live evidence includes an unattested track", ErrInvalid)
		}
	}
	expected := expectedCampaignPairedStatistics(evidence.TrackIDs)
	if len(evidence.Statistics) != len(expected) {
		return fmt.Errorf("%w: evaluation campaign paired-live statistic vector is incomplete", ErrInvalid)
	}
	for index, contract := range expected {
		if err := validateCampaignPairedStatistic(evidence.Statistics[index], contract); err != nil {
			return err
		}
	}
	if err := validateCampaignG3PromotionStatistics(evidence.PromotionStatistics); err != nil {
		return err
	}
	if err := validateCampaignArmReliability(campaign, evidence); err != nil {
		return err
	}
	digest, err := campaignPairedLiveEvidenceDigest(evidence)
	if err != nil || digest != evidence.Digest {
		return fmt.Errorf("%w: evaluation campaign paired-live evidence digest is invalid", ErrInvalid)
	}
	return nil
}

func campaignPairedProvenanceDigestsValid(evidence CampaignPairedLiveEvidence) bool {
	return digestPattern.MatchString(evidence.BaselinePolicySnapshotDigest) &&
		digestPattern.MatchString(evidence.CandidatePolicySnapshotDigest) &&
		digestPattern.MatchString(evidence.BaselineBindingSnapshotDigest) &&
		digestPattern.MatchString(evidence.CandidateBindingSnapshotDigest) &&
		digestPattern.MatchString(evidence.BaselinePoolSnapshotDigest) &&
		digestPattern.MatchString(evidence.CandidatePoolSnapshotDigest) &&
		digestPattern.MatchString(evidence.BaselineEnvironmentSnapshotDigest) &&
		digestPattern.MatchString(evidence.CandidateEnvironmentSnapshotDigest) &&
		digestPattern.MatchString(evidence.BaselineBackendTopologyDigest) &&
		digestPattern.MatchString(evidence.CandidateBackendTopologyDigest)
}

type campaignPairedStatisticContract struct {
	id, gateID, analysisUnit, direction string
	trackID                             TrackID
	margin                              float64
}

func expectedCampaignPairedStatistics(trackIDs []TrackID) []campaignPairedStatisticContract {
	contracts := make([]campaignPairedStatisticContract, 0, len(trackIDs)*3+1)
	for _, trackID := range trackIDs {
		if campaignTrackHasQualityStatistic(trackID) {
			contracts = append(contracts, campaignPairedStatisticContract{
				id: "campaign.g3." + string(trackID) + ".quality_non_inferiority", gateID: "G3",
				trackID: trackID, analysisUnit: campaignQualityAnalysisUnit(trackID),
				direction: "higher_is_better", margin: campaignQualityMargin,
			})
		}
		if trackID == "model_pool" {
			contracts = append(contracts, campaignPairedStatisticContract{
				id: "campaign.g3.model_pool.worst_arm_reliability_non_inferiority", gateID: "G3",
				trackID: trackID, analysisUnit: campaignPoolWorstArmReliabilityUnit,
				direction: "higher_is_better", margin: campaignFailureRiskMargin,
			})
		}
		contracts = append(contracts,
			campaignPairedStatisticContract{
				id: "campaign.g8." + string(trackID) + ".failure_risk", gateID: "G8",
				trackID: trackID, analysisUnit: campaignFailureAnalysisUnit(trackID),
				direction: "lower_is_better", margin: campaignFailureRiskMargin,
			},
			campaignPairedStatisticContract{
				id: "campaign.g8." + string(trackID) + ".latency_risk", gateID: "G8",
				trackID: trackID, analysisUnit: campaignLatencyUnit,
				direction: "lower_is_better", margin: campaignLatencyRiskMargin,
			},
		)
	}
	return contracts
}

func validateCampaignPairedStatistic(
	statistic CampaignPairedStatistic,
	contract campaignPairedStatisticContract,
) error {
	if statistic.ID != contract.id || statistic.GateID != contract.gateID || statistic.TrackID != contract.trackID ||
		statistic.AnalysisUnit != contract.analysisUnit || statistic.Direction != contract.direction ||
		statistic.Margin != contract.margin || statistic.ConfidenceLevel != campaignPairedConfidenceLevel ||
		statistic.SampleCount < 0 || statistic.MissingPairs < 0 || statistic.ConfidenceInterval == nil ||
		!validCampaignStatisticValues(statistic) {
		return fmt.Errorf("%w: evaluation campaign paired-live statistic is invalid", ErrInvalid)
	}
	conclusive := statistic.SampleCount >= campaignPairedMinimumCases && statistic.MissingPairs == 0
	expectsCandidateInterval := statistic.AnalysisUnit == campaignPoolWorstArmReliabilityUnit
	if (conclusive && len(statistic.ConfidenceInterval) != 2) ||
		(!conclusive && len(statistic.ConfidenceInterval) != 0) ||
		!validCampaignInterval(statistic.ConfidenceInterval) ||
		(expectsCandidateInterval && conclusive && len(statistic.CandidateConfidenceInterval) != 2) ||
		(expectsCandidateInterval && !conclusive && len(statistic.CandidateConfidenceInterval) != 0) ||
		(!expectsCandidateInterval && statistic.CandidateConfidenceInterval != nil) ||
		!validCampaignInterval(statistic.CandidateConfidenceInterval) {
		return fmt.Errorf("%w: evaluation campaign paired-live confidence decision is invalid", ErrInvalid)
	}
	expectedVerdict := campaignStatisticVerdict(statistic)
	if expectsCandidateInterval {
		expectedVerdict = campaignWorstArmReliabilityVerdict(statistic)
	}
	if statistic.Verdict != expectedVerdict {
		return fmt.Errorf("%w: evaluation campaign paired-live statistic verdict is invalid", ErrInvalid)
	}
	return nil
}

func validCampaignStatisticValues(statistic CampaignPairedStatistic) bool {
	allNil := statistic.BaselineValue == nil && statistic.CandidateValue == nil && statistic.Delta == nil
	allPresent := statistic.BaselineValue != nil && statistic.CandidateValue != nil && statistic.Delta != nil
	if (!allNil && !allPresent) || (statistic.SampleCount > 0 && statistic.MissingPairs == 0 && !allPresent) ||
		((statistic.SampleCount == 0 || statistic.MissingPairs != 0) && !allNil) {
		return false
	}
	if allNil {
		return true
	}
	baseline, candidate, delta := *statistic.BaselineValue, *statistic.CandidateValue, *statistic.Delta
	if !finiteFloat(baseline) || !finiteFloat(candidate) || !finiteFloat(delta) ||
		!reducedFloatsEqual(candidate-baseline, delta) {
		return false
	}
	switch statistic.AnalysisUnit {
	case campaignQualityUnit, campaignPoolQualityUnit, campaignPoolWorstArmReliabilityUnit,
		campaignFailureUnit, campaignPoolFailureUnit:
		return baseline >= 0 && baseline <= 1 && candidate >= 0 && candidate <= 1
	case campaignLatencyUnit:
		return baseline >= 0 && baseline <= 1 && candidate >= 0
	default:
		return false
	}
}

func validCampaignInterval(interval []float64) bool {
	return len(interval) == 0 ||
		(len(interval) == 2 && finiteFloat(interval[0]) && finiteFloat(interval[1]) && interval[0] <= interval[1])
}

func validateCampaignG3PromotionStatistics(statistics []CampaignG3PromotionStatistic) error {
	expected := campaignG3PromotionStatisticContracts()
	if len(statistics) != len(expected) {
		return fmt.Errorf("%w: G3 promotion statistic vector is incomplete", ErrInvalid)
	}
	for index, contract := range expected {
		statistic := statistics[index]
		if statistic.ID != contract.id || statistic.Direction != contract.direction ||
			statistic.Threshold != contract.threshold || statistic.ConfidenceLevel != campaignPairedConfidenceLevel ||
			statistic.SampleCount < 0 || statistic.MissingCases < 0 ||
			!finiteFloat(statistic.Estimate) || !validCampaignInterval(statistic.ConfidenceInterval) ||
			statistic.Verdict != campaignG3PromotionStatisticVerdict(statistic) {
			return fmt.Errorf("%w: G3 promotion statistic %q is invalid", ErrInvalid, statistic.ID)
		}
	}
	return nil
}

func validateCampaignArmReliability(campaign Campaign, evidence CampaignPairedLiveEvidence) error {
	poolSampleCount := campaignPromotionSampleCount(evidence.PromotionStatistics)
	if poolSampleCount <= 0 || len(evidence.ModelPoolArmReliability) < 2 {
		return fmt.Errorf("%w: model_pool campaign lacks complete frozen-arm reliability", ErrInvalid)
	}
	baselineArms, candidateArms := map[string]bool{}, map[string]bool{}
	previous := ""
	for _, statistic := range evidence.ModelPoolArmReliability {
		if !portableSuiteIDPattern.MatchString(statistic.ArmID) || statistic.ArmID <= previous ||
			statistic.Direction != "lower_is_better" || statistic.ConfidenceLevel != campaignPairedConfidenceLevel ||
			statistic.BaselineSampleCount < 0 || statistic.CandidateSampleCount < 0 ||
			statistic.ConfidenceInterval == nil || !validCampaignInterval(statistic.ConfidenceInterval) {
			return fmt.Errorf("%w: frozen-arm reliability contract is invalid", ErrInvalid)
		}
		previous = statistic.ArmID
		switch statistic.Cohort {
		case campaignArmCohortPaired:
			baselineArms[statistic.ArmID], candidateArms[statistic.ArmID] = true, true
			if statistic.Margin != campaignFailureRiskMargin || statistic.BaselineSampleCount != poolSampleCount ||
				statistic.CandidateSampleCount != poolSampleCount || len(statistic.ConfidenceInterval) != 2 ||
				len(statistic.CandidateConfidenceInterval) != 2 || !validCampaignArmFailureRates(statistic, true, true) ||
				statistic.Verdict != campaignArmReliabilityVerdict(statistic) {
				return fmt.Errorf("%w: paired frozen-arm reliability is invalid", ErrInvalid)
			}
		case campaignArmCohortCandidateOnly:
			candidateArms[statistic.ArmID] = true
			if statistic.Margin != 1-frozenCampaignG3PromotionPolicy.MinimumCandidateArmReliability ||
				statistic.BaselineSampleCount != 0 || statistic.CandidateSampleCount != poolSampleCount ||
				len(statistic.ConfidenceInterval) != 0 || len(statistic.CandidateConfidenceInterval) != 2 ||
				!validCampaignArmFailureRates(statistic, false, true) ||
				statistic.Verdict != campaignArmReliabilityVerdict(statistic) {
				return fmt.Errorf("%w: candidate-only frozen-arm reliability is invalid", ErrInvalid)
			}
		case campaignArmCohortBaselineOnly:
			baselineArms[statistic.ArmID] = true
			if statistic.Margin != campaignFailureRiskMargin || statistic.BaselineSampleCount != poolSampleCount ||
				statistic.CandidateSampleCount != 0 || !validCampaignArmFailureRates(statistic, true, false) ||
				len(statistic.ConfidenceInterval) != 0 || len(statistic.CandidateConfidenceInterval) != 0 ||
				statistic.Verdict != "unavailable" {
				return fmt.Errorf("%w: baseline-only frozen-arm disclosure is invalid", ErrInvalid)
			}
		default:
			return fmt.Errorf("%w: frozen-arm reliability cohort is invalid", ErrInvalid)
		}
	}
	if len(baselineArms) < 2 || len(candidateArms) < 2 {
		return fmt.Errorf("%w: frozen-arm reliability does not cover both pools", ErrInvalid)
	}
	changed := !reflect.DeepEqual(baselineArms, candidateArms)
	if changed && (campaign.ChangeProfile != "model_pool" ||
		evidence.BaselinePoolSnapshotDigest == evidence.CandidatePoolSnapshotDigest) {
		return fmt.Errorf("%w: frozen-arm membership changed outside model_pool treatment", ErrInvalid)
	}
	return nil
}

func validCampaignArmFailureRates(
	statistic CampaignArmReliabilityStatistic,
	expectBaseline, expectCandidate bool,
) bool {
	if (statistic.BaselineFailureRate != nil) != expectBaseline ||
		(statistic.CandidateFailureRate != nil) != expectCandidate {
		return false
	}
	if expectBaseline && (!finiteFloat(*statistic.BaselineFailureRate) ||
		*statistic.BaselineFailureRate < 0 || *statistic.BaselineFailureRate > 1) {
		return false
	}
	if expectCandidate && (!finiteFloat(*statistic.CandidateFailureRate) ||
		*statistic.CandidateFailureRate < 0 || *statistic.CandidateFailureRate > 1) {
		return false
	}
	if expectBaseline && expectCandidate {
		return statistic.Delta != nil && finiteFloat(*statistic.Delta) &&
			reducedFloatsEqual(*statistic.CandidateFailureRate-*statistic.BaselineFailureRate, *statistic.Delta)
	}
	return statistic.Delta == nil
}

func campaignPromotionSampleCount(statistics []CampaignG3PromotionStatistic) int {
	for _, statistic := range statistics {
		if statistic.ID == campaignG3CandidateNormalizedRegretID {
			return statistic.SampleCount
		}
	}
	return 0
}

func validateCampaignPairedGateBinding(
	campaign Campaign,
	anchors map[string]CampaignEvidenceAnchor,
	evidence CampaignPairedLiveEvidence,
) error {
	slot, _ := campaignSlotContract(campaign.ChangeProfile, "G3")
	base := CampaignGate{
		ID: "G3", Name: slot.Name, Disposition: slot.Disposition, Verdict: "unavailable",
		EvidenceLevel: slot.MinimumEvidenceLevel, Source: "campaign_slot", EvidenceRefs: []string{},
		Rationale: "No qualified evidence is bound to this advisory campaign slot.",
	}
	paired := campaignPairedLiveGate(
		base, evidence,
		campaignRunEvidence{anchor: anchors["g3:baseline"]},
		campaignRunEvidence{anchor: anchors["g3:candidate"]},
	)
	if !reflect.DeepEqual(campaign.Decision.Gates[3], paired) {
		return fmt.Errorf("%w: campaign G3 differs from its paired-live receipt", ErrInvalid)
	}
	return nil
}

func validateCampaignFidelityEvidence(
	campaign Campaign,
	anchors map[string]CampaignEvidenceAnchor,
	evidence CampaignFidelityEvidence,
) error {
	binding := campaign.GateBindings.G5Fidelity
	reference, live := anchors["g5:reference"], anchors["g5:live"]
	expectedTrackID, trackErr := campaignFidelityTrack(campaign.ChangeProfile)
	if binding == nil || evidence.SchemaVersion != SchemaVersion ||
		trackErr != nil || evidence.TrackID != expectedTrackID ||
		evidence.ContractVersion != CampaignFidelityContractVersion ||
		evidence.ReferenceRunID != binding.ReferenceRunID || evidence.LiveRunID != binding.LiveRunID ||
		evidence.CandidateSubjectDigest != reference.CandidateSubjectDigest ||
		evidence.CandidateSubjectDigest != live.CandidateSubjectDigest ||
		evidence.ReferenceManifestDigest != reference.ManifestSemanticDigest ||
		evidence.LiveManifestDigest != live.ManifestSemanticDigest ||
		evidence.LiveExecutionAttestationDigest != live.ExecutionAttestationDigest ||
		len(evidence.SuiteIDs) == 0 || !digestPattern.MatchString(evidence.WorkloadSnapshotDigest) ||
		len(evidence.BenchmarkRevisions) == 0 || evidence.ConfidenceLevel != fidelityConfidenceLevel ||
		evidence.SampleCount <= 0 || evidence.MatchedCases < 0 || evidence.DecisionMismatches < 0 ||
		evidence.OutcomeMismatches < 0 || evidence.UnavailableCases < 0 ||
		evidence.MatchedCases+evidence.DecisionMismatches+evidence.OutcomeMismatches+evidence.UnavailableCases != evidence.SampleCount ||
		!finiteFloat(evidence.PointEstimate) || !finiteFloat(evidence.LowerBound) ||
		!reducedFloatsEqual(evidence.PointEstimate, float64(evidence.MatchedCases)/float64(evidence.SampleCount)) ||
		!reducedFloatsEqual(evidence.LowerBound, oneSidedExactBinomialLowerBound(evidence.MatchedCases, evidence.SampleCount, fidelityConfidenceLevel)) {
		return fmt.Errorf("%w: evaluation campaign fidelity receipt is invalid", ErrInvalid)
	}
	expectedVerdict := GateVerdict("fail")
	if evidence.SampleCount < minimumFidelityCases {
		expectedVerdict = "unavailable"
	} else if evidence.LowerBound >= fidelityMinimum && evidence.UnavailableCases == 0 {
		expectedVerdict = "pass"
	}
	if evidence.Verdict != expectedVerdict {
		return fmt.Errorf("%w: evaluation campaign fidelity verdict is invalid", ErrInvalid)
	}
	digest, err := campaignFidelityEvidenceDigest(evidence)
	if err != nil || digest != evidence.Digest {
		return fmt.Errorf("%w: evaluation campaign fidelity digest is invalid", ErrInvalid)
	}
	gate := campaign.Decision.Gates[5]
	expectedRefs := append(campaignAnchorRefsForKeys(anchors, "g5:reference", "g5:live"), "campaign-fidelity:"+evidence.Digest)
	if gate.Verdict != evidence.Verdict || gate.EvidenceLevel != "E5" || gate.Source != "reference_to_fresh_live" ||
		gate.Observed == nil || !reducedFloatsEqual(*gate.Observed, evidence.LowerBound) ||
		gate.Threshold == nil || gate.Threshold.Operator != ">=" || gate.Threshold.Value != fidelityMinimum ||
		gate.Threshold.Unit != "fraction" || gate.SampleCount != evidence.SampleCount ||
		!reflect.DeepEqual(gate.EvidenceRefs, expectedRefs) {
		return fmt.Errorf("%w: campaign G5 differs from its fidelity receipt", ErrInvalid)
	}
	return nil
}
