package evaluationplane

import (
	"fmt"
	"reflect"
	"sort"
)

func alignCampaignAttestedObservations(
	baseline []campaignAttestedObservation,
	candidate []campaignAttestedObservation,
) ([]campaignObservationPair, error) {
	baselineByID := make(map[campaignObservationCoordinate]campaignAttestedObservation, len(baseline))
	candidateByID := make(map[campaignObservationCoordinate]campaignAttestedObservation, len(candidate))
	for _, record := range baseline {
		coordinate := campaignObservationAnalysisCoordinate(record)
		if _, duplicate := baselineByID[coordinate]; duplicate {
			return nil, fmt.Errorf("%w: baseline live records duplicate an analysis identity", ErrInvalid)
		}
		baselineByID[coordinate] = record
	}
	for _, record := range candidate {
		coordinate := campaignObservationAnalysisCoordinate(record)
		if _, duplicate := candidateByID[coordinate]; duplicate {
			return nil, fmt.Errorf("%w: candidate live records duplicate an analysis identity", ErrInvalid)
		}
		candidateByID[coordinate] = record
	}
	if len(baselineByID) == 0 || len(baselineByID) != len(candidateByID) {
		return nil, fmt.Errorf("%w: paired live records are not exactly aligned", ErrInvalid)
	}
	coordinates := make([]campaignObservationCoordinate, 0, len(baselineByID))
	for coordinate := range baselineByID {
		coordinates = append(coordinates, coordinate)
	}
	sort.Slice(coordinates, func(left, right int) bool {
		return coordinates[left].sortKey() < coordinates[right].sortKey()
	})
	pairs := make([]campaignObservationPair, 0, len(coordinates))
	for _, coordinate := range coordinates {
		old := baselineByID[coordinate]
		current, ok := candidateByID[coordinate]
		if !ok {
			return nil, fmt.Errorf("%w: paired live record analysis identities do not match", ErrInvalid)
		}
		pairs = append(pairs, campaignObservationPair{baseline: old, candidate: current})
	}
	return pairs, nil
}

type campaignObservationCoordinate struct {
	trackID        TrackID
	caseID         string
	attemptID      string
	operation      string
	armID          string
	concurrency    int64
	hasConcurrency bool
	modality       string
	hasModality    bool
	loadPhase      string
	hasLoadPhase   bool
	loadRepeat     int64
	hasLoadRepeat  bool
	loadIndex      int64
	hasLoadIndex   bool
}

func campaignObservationAnalysisCoordinate(observation campaignAttestedObservation) campaignObservationCoordinate {
	coordinate := campaignObservationCoordinate{
		trackID: observation.trackID, caseID: observation.caseID,
		attemptID: observation.attemptID, operation: observation.operation,
	}
	// A direct model-pool arm is a treatment coordinate. A selected arm from
	// routing or joint is an outcome and therefore must not enter the pairing
	// identity: a candidate recipe is allowed to make a different decision for
	// the same case.
	if observation.trackID == "model_pool" {
		coordinate.armID = observation.armID
	}
	if observation.concurrency != nil {
		coordinate.concurrency, coordinate.hasConcurrency = *observation.concurrency, true
	}
	if observation.modality != nil {
		coordinate.modality, coordinate.hasModality = *observation.modality, true
	}
	if observation.loadPhase != nil {
		coordinate.loadPhase, coordinate.hasLoadPhase = *observation.loadPhase, true
	}
	if observation.loadRepeat != nil {
		coordinate.loadRepeat, coordinate.hasLoadRepeat = *observation.loadRepeat, true
	}
	if observation.loadIndex != nil {
		coordinate.loadIndex, coordinate.hasLoadIndex = *observation.loadIndex, true
	}
	return coordinate
}

func (coordinate campaignObservationCoordinate) sortKey() string {
	return fmt.Sprintf(
		"%s\x00%s\x00%s\x00%s\x00%s\x00%t:%d\x00%t:%s\x00%t:%s\x00%t:%d\x00%t:%d",
		coordinate.trackID, coordinate.caseID, coordinate.attemptID, coordinate.operation, coordinate.armID,
		coordinate.hasConcurrency, coordinate.concurrency, coordinate.hasModality, coordinate.modality,
		coordinate.hasLoadPhase, coordinate.loadPhase, coordinate.hasLoadRepeat, coordinate.loadRepeat,
		coordinate.hasLoadIndex, coordinate.loadIndex,
	)
}

func alignCampaignPairedLiveObservations(
	baseline []campaignAttestedObservation,
	candidate []campaignAttestedObservation,
	profile ChangeProfile,
	baselineMixture *CatalogMixture,
	candidateMixture *CatalogMixture,
	baselinePoolDigest string,
	candidatePoolDigest string,
) (campaignPairedCohort, error) {
	baselinePool, baselineOther := partitionCampaignPoolObservations(baseline)
	candidatePool, candidateOther := partitionCampaignPoolObservations(candidate)
	var exactPairs []campaignObservationPair
	if len(baselineOther) != 0 || len(candidateOther) != 0 {
		var err error
		exactPairs, err = alignCampaignAttestedObservations(baselineOther, candidateOther)
		if err != nil {
			return campaignPairedCohort{}, err
		}
	}
	if len(baselinePool) == 0 && len(candidatePool) == 0 {
		return campaignPairedCohort{exactPairs: exactPairs}, nil
	}
	if baselineMixture == nil || candidateMixture == nil {
		return campaignPairedCohort{}, fmt.Errorf("%w: model_pool paired evidence requires both frozen mixtures", ErrInvalid)
	}
	baselineArms := campaignMixtureArmIDs(baselineMixture)
	candidateArms := campaignMixtureArmIDs(candidateMixture)
	if len(baselineArms) == 0 || len(candidateArms) == 0 {
		return campaignPairedCohort{}, fmt.Errorf("%w: model_pool paired evidence has an empty frozen pool", ErrInvalid)
	}
	armSetsMatch := reflect.DeepEqual(baselineArms, candidateArms)
	if armSetsMatch {
		if _, err := alignCampaignAttestedObservations(baselinePool, candidatePool); err != nil {
			return campaignPairedCohort{}, fmt.Errorf("%w: model_pool arm coordinates are not exactly aligned", ErrInvalid)
		}
	} else if profile != "model_pool" || baselinePoolDigest == candidatePoolDigest {
		return campaignPairedCohort{}, fmt.Errorf(
			"%w: model_pool arm membership may differ only for a declared model_pool treatment with a changed pool snapshot",
			ErrInvalid,
		)
	}
	poolCases, err := alignCampaignPoolCases(
		baselinePool, candidatePool, baselineArms, candidateArms,
	)
	if err != nil {
		return campaignPairedCohort{}, err
	}
	return campaignPairedCohort{exactPairs: exactPairs, poolCases: poolCases}, nil
}

func partitionCampaignPoolObservations(
	observations []campaignAttestedObservation,
) (pool []campaignAttestedObservation, other []campaignAttestedObservation) {
	pool = make([]campaignAttestedObservation, 0)
	other = make([]campaignAttestedObservation, 0, len(observations))
	for _, observation := range observations {
		if observation.trackID == "model_pool" {
			pool = append(pool, observation)
		} else {
			other = append(other, observation)
		}
	}
	return pool, other
}

func campaignMixtureArmIDs(mixture *CatalogMixture) []string {
	armIDs := make([]string, 0, len(mixture.ModelArms))
	for _, arm := range mixture.ModelArms {
		armIDs = append(armIDs, arm.ID)
	}
	sort.Strings(armIDs)
	return armIDs
}

func alignCampaignPoolCases(
	baseline []campaignAttestedObservation,
	candidate []campaignAttestedObservation,
	baselineArms []string,
	candidateArms []string,
) ([]campaignPoolCasePair, error) {
	baselineByCase, err := campaignDensePoolCases("baseline_live", baseline, baselineArms)
	if err != nil {
		return nil, err
	}
	candidateByCase, err := campaignDensePoolCases("candidate_live", candidate, candidateArms)
	if err != nil {
		return nil, err
	}
	if len(baselineByCase) == 0 || len(baselineByCase) != len(candidateByCase) {
		return nil, fmt.Errorf("%w: model_pool case cohorts are not exactly aligned", ErrInvalid)
	}
	caseIDs := sortedMapKeys(baselineByCase)
	pairs := make([]campaignPoolCasePair, 0, len(caseIDs))
	for _, caseID := range caseIDs {
		candidateRows, present := candidateByCase[caseID]
		if !present {
			return nil, fmt.Errorf("%w: model_pool case cohorts are not exactly aligned", ErrInvalid)
		}
		if err := validateCampaignSharedArmCoordinates(baselineByCase[caseID], candidateRows); err != nil {
			return nil, err
		}
		pairs = append(pairs, campaignPoolCasePair{
			caseID: caseID, baseline: baselineByCase[caseID], candidate: candidateRows,
		})
	}
	return pairs, nil
}

func validateCampaignSharedArmCoordinates(
	baseline []campaignAttestedObservation,
	candidate []campaignAttestedObservation,
) error {
	baselineByArm := make(map[string]campaignAttestedObservation, len(baseline))
	for _, observation := range baseline {
		baselineByArm[observation.armID] = observation
	}
	for _, observation := range candidate {
		old, shared := baselineByArm[observation.armID]
		if !shared {
			continue
		}
		if campaignObservationAnalysisCoordinate(old) != campaignObservationAnalysisCoordinate(observation) {
			return fmt.Errorf(
				"%w: model_pool shared arm %q is not aligned on the same case and attempt",
				ErrInvalid,
				observation.armID,
			)
		}
	}
	return nil
}

func campaignDensePoolCases(
	role string,
	observations []campaignAttestedObservation,
	expectedArms []string,
) (map[string][]campaignAttestedObservation, error) {
	expected := make(map[string]struct{}, len(expectedArms))
	for _, armID := range expectedArms {
		expected[armID] = struct{}{}
	}
	byCase := make(map[string][]campaignAttestedObservation)
	seen := make(map[string]map[string]bool)
	for _, observation := range observations {
		if observation.operation != workerBrokerArmChatCompletion || observation.armID == "" {
			return nil, fmt.Errorf("%w: %s model_pool observation is not a direct frozen-arm operation", ErrInvalid, role)
		}
		if _, present := expected[observation.armID]; !present {
			return nil, fmt.Errorf("%w: %s model_pool observation names an arm outside its frozen pool", ErrInvalid, role)
		}
		if seen[observation.caseID] == nil {
			seen[observation.caseID] = make(map[string]bool, len(expected))
		}
		if seen[observation.caseID][observation.armID] {
			return nil, fmt.Errorf("%w: %s model_pool repeats a case/arm coordinate", ErrInvalid, role)
		}
		seen[observation.caseID][observation.armID] = true
		byCase[observation.caseID] = append(byCase[observation.caseID], observation)
	}
	for caseID, arms := range seen {
		if len(arms) != len(expected) {
			return nil, fmt.Errorf("%w: %s model_pool case %q is not dense over its frozen arms", ErrInvalid, role, caseID)
		}
		sort.Slice(byCase[caseID], func(left, right int) bool {
			return byCase[caseID][left].armID < byCase[caseID][right].armID
		})
	}
	return byCase, nil
}
