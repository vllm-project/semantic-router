package evaluationplane

import (
	"fmt"
	"time"
)

func buildCampaignDecision(
	campaign Campaign,
	evidence map[string]campaignRunEvidence,
	now time.Time,
) (CampaignDecision, error) {
	var paired *CampaignPairedLiveEvidence
	var fidelity *CampaignFidelityEvidence
	var err error
	if campaign.GateBindings.G3ControlledPair != nil {
		paired, err = buildCampaignPairedLiveEvidence(
			evidence[campaignEvidenceKey("g3", "baseline")],
			evidence[campaignEvidenceKey("g3", "candidate")],
		)
		if err != nil {
			return CampaignDecision{}, err
		}
	}
	if campaign.GateBindings.G5Fidelity != nil {
		fidelity, err = buildCampaignFidelityEvidence(
			evidence[campaignEvidenceKey("g5", "reference")],
			evidence[campaignEvidenceKey("g5", "live")],
		)
		if err != nil {
			return CampaignDecision{}, err
		}
	}
	gates := make([]CampaignGate, 0, len(requiredGateIDs))
	for _, gateID := range requiredGateIDs {
		gates = append(gates, campaignGateFor(
			campaign.ChangeProfile, campaign.GateBindings, gateID, evidence, paired, fidelity,
		))
	}
	bindings, err := campaignEvidenceBindings(campaign.GateBindings)
	if err != nil {
		return CampaignDecision{}, err
	}
	anchors := make([]CampaignEvidenceAnchor, 0, len(bindings))
	for _, binding := range bindings {
		anchors = append(anchors, evidence[campaignEvidenceKey(binding.slotID, binding.bindingRole)].anchor)
	}
	verdict, summary, recommendations := campaignDecisionSummary(gates)
	decision := CampaignDecision{
		SchemaVersion: SchemaVersion, ContractVersion: CampaignContractVersion,
		AttestationRevision: ServerAttestationRevision, CampaignID: campaign.ID,
		CampaignDigest: campaign.ManifestDigest, Verdict: verdict, Summary: summary,
		Gates: gates, Evidence: anchors, PairedLiveEvidence: paired, FidelityEvidence: fidelity,
		Recommendations: recommendations, CreatedAt: now,
	}
	decision.DecisionDigest, err = campaignDecisionDigest(decision)
	if err != nil {
		return CampaignDecision{}, err
	}
	return decision, nil
}

func campaignGateFor(
	profile ChangeProfile,
	bindings CampaignGateBindings,
	gateID string,
	evidence map[string]campaignRunEvidence,
	paired *CampaignPairedLiveEvidence,
	fidelity *CampaignFidelityEvidence,
) CampaignGate {
	definition, _ := releaseGateDefinitionByID(gateID)
	disposition, _ := releaseProfileDisposition(profile, gateID)
	if gateID == "G0" || gateID == "G1" {
		return CampaignGate{
			ID: gateID, Name: definition.Name, Disposition: disposition, Verdict: "pass",
			EvidenceLevel: "E5", Source: "server_anchors",
			EvidenceRefs: campaignEvidenceRefs(bindings, evidence), Observed: float64Reference(1),
			Threshold: &GateThreshold{Operator: ">=", Value: 1, Unit: "boolean"},
			Rationale: "Every bound run has an immutable manifest, sealed private receipt, server report anchor, and exact slot identity.",
		}
	}
	slot, _ := campaignSlotContract(profile, gateID)
	base := CampaignGate{
		ID: gateID, Name: slot.Name, Disposition: slot.Disposition,
		Verdict: "unavailable", EvidenceLevel: slot.MinimumEvidenceLevel,
		Source: "campaign_slot", EvidenceRefs: []string{},
		Rationale: "No qualified evidence is bound to this advisory campaign slot.",
	}
	if slot.Disposition == GateDispositionNotApplicable {
		base.Verdict, base.Source = "not_applicable", "campaign_contract"
		base.Rationale = "The gate is not applicable to this change profile."
		return base
	}
	slotID := "g" + gateID[1:]
	switch gateID {
	case "G3":
		if paired == nil {
			return base
		}
		return campaignPairedLiveGate(
			base, *paired,
			evidence[campaignEvidenceKey("g3", "baseline")],
			evidence[campaignEvidenceKey("g3", "candidate")],
		)
	case "G5":
		if fidelity == nil {
			return base
		}
		base.Verdict, base.EvidenceLevel, base.Source = fidelity.Verdict, "E5", "reference_to_fresh_live"
		base.Observed = float64Reference(fidelity.LowerBound)
		base.Threshold = &GateThreshold{Operator: ">=", Value: fidelityMinimum, Unit: "fraction"}
		base.SampleCount = fidelity.SampleCount
		base.EvidenceRefs = append(
			campaignEvidenceRefsFor(evidence, campaignEvidenceKey("g5", "reference"), campaignEvidenceKey("g5", "live")),
			"campaign-fidelity:"+fidelity.Digest,
		)
		base.Rationale = fmt.Sprintf(
			"Reference-to-fresh-live agreement: %d/%d cases matched, with %d decision, %d outcome, and %d unavailable case(s); one-sided 95%% lower bound %.6g.",
			fidelity.MatchedCases, fidelity.SampleCount, fidelity.DecisionMismatches,
			fidelity.OutcomeMismatches, fidelity.UnavailableCases, fidelity.LowerBound,
		)
		return base
	default:
		item, found := evidence[campaignEvidenceKey(slotID, campaignSingleBindingRole)]
		if !found {
			return base
		}
		gate, found := reportGate(item.report, gateID)
		if !found {
			return base
		}
		return reportCampaignGate(base, gate, "campaign_slot:"+slotID, item.anchor)
	}
}

func reportGate(report Report, id string) (Gate, bool) {
	return reportGateFromSlice(report.Gates, id)
}

func reportGateFromSlice(gates []Gate, id string) (Gate, bool) {
	for _, gate := range gates {
		if gate.ID == id {
			return gate, true
		}
	}
	return Gate{}, false
}

func reportCampaignGate(base CampaignGate, source Gate, owner string, anchor CampaignEvidenceAnchor) CampaignGate {
	base.Verdict, base.EvidenceLevel, base.Source = source.Verdict, source.EvidenceLevel, owner
	base.Observed, base.Threshold, base.Rationale = source.Observed, source.Threshold, source.Rationale
	if source.SampleCount != nil {
		base.SampleCount = *source.SampleCount
	}
	base.EvidenceRefs = anchorEvidenceRefs(anchor)
	return base
}

func anchorEvidenceRefs(anchor CampaignEvidenceAnchor) []string {
	refs := []string{
		"run:" + anchor.SlotID + ":" + anchor.BindingRole + ":" + anchor.RunID,
		"manifest-semantic:" + anchor.ManifestSemanticDigest,
		"manifest-artifact:" + anchor.ManifestArtifactDigest,
		"report:" + anchor.ReportDigest,
		"private-receipt:" + anchor.PrivateReceiptDigest,
	}
	if anchor.CandidateSubjectDigest != "" {
		refs = append(refs, "candidate-subject:"+anchor.CandidateSubjectDigest)
	}
	if anchor.ExecutionAttestationDigest != "" {
		refs = append(refs, "execution-attestation:"+anchor.ExecutionAttestationDigest)
	}
	return refs
}

func campaignEvidenceRefsFor(evidence map[string]campaignRunEvidence, keys ...string) []string {
	refs := make([]string, 0, len(keys)*6)
	for _, key := range keys {
		if item, ok := evidence[key]; ok {
			refs = append(refs, anchorEvidenceRefs(item.anchor)...)
		}
	}
	return refs
}

func campaignEvidenceRefs(
	gateBindings CampaignGateBindings,
	evidence map[string]campaignRunEvidence,
) []string {
	bindings, _ := campaignEvidenceBindings(gateBindings)
	refs := make([]string, 0, len(bindings)*6)
	for _, binding := range bindings {
		refs = append(refs, anchorEvidenceRefs(evidence[campaignEvidenceKey(binding.slotID, binding.bindingRole)].anchor)...)
	}
	return refs
}

func campaignDecisionSummary(gates []CampaignGate) (DecisionVerdict, string, []string) {
	failed, unavailable, passed := 0, 0, 0
	for _, gate := range gates {
		if gate.Disposition != GateDispositionRequired {
			continue
		}
		switch gate.Verdict {
		case "fail":
			failed++
		case "pass":
			passed++
		default:
			unavailable++
		}
	}
	if failed > 0 {
		return DecisionVerdictFail, fmt.Sprintf("Promotion blocked: %d required gate(s) failed; %d passed.", failed, passed),
			[]string{"Keep the baseline active and resolve every failed campaign gate before another decision."}
	}
	if unavailable > 0 {
		return DecisionVerdictUnavailable, fmt.Sprintf("Decision incomplete: %d required gate(s) passed and %d lack qualified evidence.", passed, unavailable),
			[]string{"Produce qualified evidence for every required campaign slot; unavailable evidence is never inferred as pass."}
	}
	return DecisionVerdictPass, fmt.Sprintf("Promotion qualified: all %d required campaign gates passed.", passed),
		[]string{"Proceed through the declared rollout controls while retaining the sealed baseline and campaign anchors."}
}
