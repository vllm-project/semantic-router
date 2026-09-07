package evaluationplane

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"strings"
)

var sourceRevisionPattern = regexp.MustCompile("^(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})$")

func (s *Service) CreateRunAs(ctx context.Context, actor Actor, request CreateRunRequest) (Run, error) {
	release, operationErr := s.beginOperation()
	if operationErr != nil {
		return Run{}, operationErr
	}
	defer release()
	if err := validateActor(actor); err != nil {
		return Run{}, err
	}
	registry, registryErr := s.registrySnapshot()
	if registryErr != nil {
		return Run{}, registryErr
	}
	validated, target, requestErr := s.validateCreateRequest(registry, request)
	if requestErr != nil {
		return Run{}, requestErr
	}
	evidenceLevel, evidenceErr := selectedSuiteEvidenceLevel(registry, validated.SuiteIDs, validated.Mode)
	if evidenceErr != nil {
		return Run{}, evidenceErr
	}
	if qualificationErr := requireQualifiedCodeRevision(evidenceLevel, s.codeRevision); qualificationErr != nil {
		return Run{}, qualificationErr
	}
	if existing, getErr := s.store.getRunForCreateRetry(validated.ClientRequestID); getErr == nil {
		return s.resolveExistingCreate(actor, validated, existing.ID)
	} else if !errors.Is(getErr, ErrNotFound) {
		return Run{}, getErr
	}
	if baselineErr := s.validateCreateBaseline(actor, validated, target); baselineErr != nil {
		return Run{}, baselineErr
	}
	if validated.BaselineRunID != "" {
		// Authorize and classify the requested baseline before exposing global
		// ledger health. Publication still requires the complete ledger and
		// revalidates the same reference under the mutation lock boundary.
		if ledgerErr := s.requireCompleteRunLedger(); ledgerErr != nil {
			return Run{}, ledgerErr
		}
	}
	run, manifest, err := s.newPendingRunManifest(registry, validated, target, evidenceLevel)
	if err != nil {
		return Run{}, err
	}
	return s.persistPendingRunAs(actor, validated, run, manifest)
}

func requireQualifiedCodeRevision(_ EvidenceLevel, revision string) error {
	if !sourceRevisionPattern.MatchString(strings.TrimSpace(revision)) {
		return fmt.Errorf("%w: evaluation requires a full Git commit or sha256 source-tree revision", ErrInvalid)
	}
	return nil
}

func validateComparableRunRequest(candidate CreateRunRequest, baseline Run) error {
	if candidate.Mode != baseline.Mode || candidate.TargetID != baseline.TargetID ||
		candidate.ChangeProfile != baseline.ChangeProfile ||
		candidate.SampleLimit != baseline.SampleLimit || candidate.Concurrency != baseline.Concurrency || candidate.Seed != baseline.Seed ||
		!reflect.DeepEqual(candidate.CapacitySLO, baseline.CapacitySLO) ||
		!reflect.DeepEqual(candidate.CapacityLoadProtocol, baseline.CapacityLoadProtocol) ||
		!reflect.DeepEqual(candidate.SuiteIDs, baseline.SuiteIDs) || !reflect.DeepEqual(candidate.TrackIDs, baseline.TrackIDs) {
		return fmt.Errorf("%w: candidate change_profile, mode, target, suites, tracks, sample_limit, concurrency, capacity_slo, and seed must match the baseline", ErrInvalid)
	}
	return nil
}

func (s *Service) validateComparableTargetSnapshot(
	profile ChangeProfile,
	target targetDefinition,
	baselineRunID string,
) error {
	baseline, _, err := s.readDurableManifest(baselineRunID)
	if err != nil {
		return fmt.Errorf("%w: baseline manifest is unavailable", ErrInvalid)
	}
	if baseline.ChangeProfile != profile {
		return fmt.Errorf("%w: baseline manifest change_profile does not match", ErrInvalid)
	}
	allowed := comparisonTreatment(profile)
	if !allowed.supported {
		return fmt.Errorf(
			"%w: change_profile %q has no independent server-owned treatment factor and cannot be paired",
			ErrInvalid, profile,
		)
	}
	codeChanged := baseline.CodeRevision != s.codeRevision
	if codeChanged && !allowed.code {
		return fmt.Errorf("%w: source code revision must remain frozen for change_profile %q", ErrInvalid, profile)
	}
	poolChanged := !sameMixturePool(baseline.Target.Mixture, target.Mixture)
	if poolChanged && !allowed.pool {
		return fmt.Errorf("%w: model pool snapshot must remain frozen for change_profile %q", ErrInvalid, profile)
	}
	bindingChanged := !sameMixtureBinding(baseline.Target.Mixture, target.Mixture)
	if bindingChanged && !allowed.binding {
		return fmt.Errorf("%w: candidate binding snapshot must remain frozen for change_profile %q", ErrInvalid, profile)
	}
	selectorChanged, selectorAvailable := changedMixtureDigest(
		baseline.Target.Mixture, target.Mixture, func(mixture *ManifestMixture) string { return mixture.SelectorDigest },
	)
	if selectorChanged && !allowed.selector {
		return fmt.Errorf("%w: selector snapshot must remain frozen for change_profile %q", ErrInvalid, profile)
	}
	adaptationChanged, adaptationAvailable := changedMixtureDigest(
		baseline.Target.Mixture, target.Mixture, func(mixture *ManifestMixture) string { return mixture.AdaptationDigest },
	)
	if adaptationChanged && !allowed.adaptation {
		return fmt.Errorf("%w: online adaptation snapshot must remain frozen for change_profile %q", ErrInvalid, profile)
	}
	productionEnvironmentChanged := baseline.Target.RouterAPIURL != target.RouterAPIURL ||
		baseline.Target.EnvoyURL != target.EnvoyURL ||
		!reflect.DeepEqual(baseline.Target.RouterAPIKey, target.RouterAPIKey) ||
		!reflect.DeepEqual(baseline.Target.EnvoyAPIKey, target.EnvoyAPIKey) ||
		!reflect.DeepEqual(baseline.Target.AgentTaskLedger, target.AgentTaskLedger) ||
		!reflect.DeepEqual(baseline.Target.FaultRecoveryLedger, target.FaultRecoveryLedger) ||
		!reflect.DeepEqual(baseline.Target.HardPolicyLedger, target.HardPolicyLedger) ||
		!reflect.DeepEqual(baseline.Target.ProductionExperimentLedger, target.ProductionExperimentLedger)
	if profile == "model_pool" && productionEnvironmentChanged {
		return fmt.Errorf("%w: model_pool treatment must freeze runtime origins, credentials, and production ledgers", ErrInvalid)
	}
	environmentChanged := baseline.Target.BackendTopologyDigest != target.BackendTopologyDigest || productionEnvironmentChanged
	if !allowed.environment && environmentChanged {
		return fmt.Errorf("%w: runtime environment snapshot must remain frozen for change_profile %q", ErrInvalid, profile)
	}
	candidatePolicyDigest := policySnapshotDigestForTarget(target, baseline.SuiteRevisions)
	if !digestPattern.MatchString(baseline.PolicySnapshotDigest) || !digestPattern.MatchString(candidatePolicyDigest) {
		return fmt.Errorf("%w: policy snapshot identity is unavailable", ErrInvalid)
	}
	policyChanged := baseline.PolicySnapshotDigest != candidatePolicyDigest
	if policyChanged && !allowed.policy {
		return fmt.Errorf("%w: policy snapshot must remain frozen for change_profile %q", ErrInvalid, profile)
	}
	primaryChanged := map[string]bool{
		"code": codeChanged, "policy": policyChanged, "selector": selectorChanged,
		"adaptation": adaptationChanged, "binding": bindingChanged,
		"pool": poolChanged, "environment": environmentChanged,
	}[allowed.primary]
	// Recorded targets do not expose a server-owned live Mixture factor graph at
	// create time. Preserve cohort/freeze checks here and require the exact
	// treatment from the sealed report factors before any comparison can exist.
	if baseline.Target.Mixture == nil && target.Mixture == nil && allowed.primary != "code" {
		return nil
	}
	if (allowed.primary == "selector" && !selectorAvailable) ||
		(allowed.primary == "adaptation" && !adaptationAvailable) {
		return fmt.Errorf(
			"%w: change_profile %q requires a server-owned %s snapshot",
			ErrInvalid, profile, allowed.primary,
		)
	}
	if !primaryChanged {
		return fmt.Errorf(
			"%w: change_profile %q requires the %s treatment factor to change",
			ErrInvalid, profile, allowed.primary,
		)
	}
	return nil
}

func sameMixtureBinding(left, right *ManifestMixture) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return left.ID == right.ID && left.BindingDigest == right.BindingDigest
}

func changedMixtureDigest(
	left, right *ManifestMixture,
	value func(*ManifestMixture) string,
) (bool, bool) {
	if left == nil || right == nil {
		return left != right, false
	}
	return value(left) != value(right), true
}
