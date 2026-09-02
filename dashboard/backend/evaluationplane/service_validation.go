package evaluationplane

import (
	"fmt"
	"sort"
	"strings"

	"github.com/google/uuid"
)

const (
	maxRunNameLength        = 200
	maxRunDescriptionLength = 4000
	maxSampleLimit          = 100000
	maxRunConcurrency       = 128
)

func (s *Service) validateCreateRequest(registry *Registry, request CreateRunRequest) (CreateRunRequest, targetDefinition, error) {
	request, normalizeErr := normalizeCreateRunRequest(request)
	if normalizeErr != nil {
		return request, targetDefinition{}, normalizeErr
	}
	if _, ok := registry.changeProfile(request.ChangeProfile); !ok {
		return request, targetDefinition{}, fmt.Errorf("%w: unknown change_profile", ErrInvalid)
	}
	target, targetErr := resolveCreateRunTarget(registry, request)
	if targetErr != nil {
		return request, targetDefinition{}, targetErr
	}
	selectedSuiteTracks, executorErr := resolveCreateRunExecutor(registry, request, target)
	if executorErr != nil {
		return request, targetDefinition{}, executorErr
	}
	if err := validateCreateRunTracks(registry, request, target, selectedSuiteTracks); err != nil {
		return request, targetDefinition{}, err
	}
	if err := validateAgenticSuiteEndpoints(
		request.Mode, request.SuiteIDs, target.AgentTaskLedger, target.FaultRecoveryLedger,
	); err != nil {
		return request, targetDefinition{}, err
	}
	request, finalizeErr := finalizeCreateRunRequest(request)
	if finalizeErr != nil {
		return request, targetDefinition{}, finalizeErr
	}
	return request, target, nil
}

func normalizeCreateRunRequest(request CreateRunRequest) (CreateRunRequest, error) {
	if request.Name != strings.TrimSpace(request.Name) ||
		request.ClientRequestID != strings.TrimSpace(request.ClientRequestID) ||
		request.Description != strings.TrimSpace(request.Description) ||
		request.TargetID != strings.TrimSpace(request.TargetID) ||
		string(request.ChangeProfile) != strings.TrimSpace(string(request.ChangeProfile)) ||
		request.BaselineRunID != strings.TrimSpace(request.BaselineRunID) {
		return request, fmt.Errorf("%w: create run scalar fields must already be trimmed", ErrInvalid)
	}
	var err error
	request.SuiteIDs, err = strictSuiteIDs(request.SuiteIDs)
	if err != nil {
		return request, err
	}
	request.TrackIDs, err = strictTrackIDs(request.TrackIDs)
	if err != nil {
		return request, err
	}
	parsed, err := uuid.Parse(request.ClientRequestID)
	if err != nil || parsed.String() != request.ClientRequestID {
		return request, fmt.Errorf("%w: client_request_id must be a canonical UUID", ErrInvalid)
	}
	if request.Name == "" || len(request.Name) > maxRunNameLength {
		return request, fmt.Errorf("%w: name must be 1-%d characters", ErrInvalid, maxRunNameLength)
	}
	if len(request.Description) > maxRunDescriptionLength {
		return request, fmt.Errorf("%w: description exceeds %d characters", ErrInvalid, maxRunDescriptionLength)
	}
	if request.Mode != ModeReplay && request.Mode != ModeLive {
		return request, fmt.Errorf("%w: mode must be replay or live", ErrInvalid)
	}
	if request.SampleLimit < 1 || request.SampleLimit > maxSampleLimit {
		return request, fmt.Errorf("%w: sample_limit must be between 1 and %d", ErrInvalid, maxSampleLimit)
	}
	if request.Concurrency < 1 || request.Concurrency > maxRunConcurrency {
		return request, fmt.Errorf("%w: concurrency must be between 1 and %d", ErrInvalid, maxRunConcurrency)
	}
	if request.Seed < 0 || request.Seed > 1<<32-1 {
		return request, fmt.Errorf("%w: seed must be between 0 and %d", ErrInvalid, uint64(1<<32-1))
	}
	if len(request.SuiteIDs) == 0 || len(request.TrackIDs) == 0 {
		return request, fmt.Errorf("%w: at least one suite and track are required", ErrInvalid)
	}
	return request, nil
}

func resolveCreateRunTarget(registry *Registry, request CreateRunRequest) (targetDefinition, error) {
	target, ok := registry.target(request.TargetID)
	if !ok {
		return targetDefinition{}, fmt.Errorf("%w: unknown target_id", ErrInvalid)
	}
	if !containsMode(target.Public.Modes, request.Mode) {
		return targetDefinition{}, fmt.Errorf("%w: target does not support requested mode", ErrInvalid)
	}
	if request.Mode == ModeLive && !digestPattern.MatchString(target.BackendTopologyDigest) {
		return targetDefinition{}, fmt.Errorf("%w: live target backend topology identity is unavailable", ErrInvalid)
	}
	return target, nil
}

func resolveCreateRunExecutor(
	registry *Registry,
	request CreateRunRequest,
	target targetDefinition,
) (map[TrackID]bool, error) {
	selectedSuiteTracks := make(map[TrackID]bool)
	selectedExecutor := ""
	for _, suiteID := range request.SuiteIDs {
		suite, ok := registry.suite(suiteID)
		if !ok {
			return nil, fmt.Errorf("%w: unknown suite %q", ErrInvalid, suiteID)
		}
		executor, executable := suiteExecutorForMode(suite, request.Mode)
		if !executable {
			return nil, fmt.Errorf("%w: suite %q does not support requested mode", ErrInvalid, suiteID)
		}
		if selectedExecutor == "" {
			selectedExecutor = executor
		} else if selectedExecutor != executor {
			return nil, fmt.Errorf("%w: one run cannot mix suite executor identities", ErrInvalid)
		}
		var liveTracks map[TrackID]struct{}
		if request.Mode == ModeLive && executor == normalizedSuiteLiveExecutorID {
			liveTracks = normalizedSuiteLiveMethodTracks(suite)
			for _, trackID := range request.TrackIDs {
				if _, admitted := liveTracks[trackID]; containsTrack(suite.TrackIDs, trackID) && !admitted {
					return nil, fmt.Errorf("%w: suite %q has no first-party normalized-live method for track %q", ErrInvalid, suiteID, trackID)
				}
			}
		}
		for _, trackID := range suite.TrackIDs {
			if liveTracks == nil {
				selectedSuiteTracks[trackID] = true
			} else if _, admitted := liveTracks[trackID]; admitted {
				selectedSuiteTracks[trackID] = true
			}
		}
	}
	contract, registered := registry.executor(selectedExecutor)
	if !registered || contract.Mode != request.Mode {
		return nil, fmt.Errorf("%w: suite executor is not registered for requested mode", ErrInvalid)
	}
	if !executorTargetMatches(selectedExecutor, request.Mode, target.Public) {
		return nil, fmt.Errorf("%w: suite execution strategy does not match target", ErrInvalid)
	}
	if contract.CaseBudgetPerSuite {
		if len(request.SuiteIDs) > maxRecordsPerRun/request.SampleLimit {
			return nil, fmt.Errorf("%w: composed normalized suite case budget exceeds %d", ErrInvalid, maxRecordsPerRun)
		}
	}
	return selectedSuiteTracks, nil
}

func validateCreateRunTracks(
	registry *Registry,
	request CreateRunRequest,
	target targetDefinition,
	selectedSuiteTracks map[TrackID]bool,
) error {
	for _, trackID := range request.TrackIDs {
		track, ok := registry.track(trackID)
		if !ok {
			return fmt.Errorf("%w: unknown track %q", ErrInvalid, trackID)
		}
		if !selectedSuiteTracks[trackID] {
			return fmt.Errorf("%w: track %q is not provided by selected suites", ErrInvalid, trackID)
		}
		if !containsMode(track.Modes, request.Mode) || !containsTrack(target.Public.TrackIDs, trackID) {
			return fmt.Errorf("%w: target cannot execute track %q in requested mode", ErrInvalid, trackID)
		}
	}
	return nil
}

func finalizeCreateRunRequest(request CreateRunRequest) (CreateRunRequest, error) {
	request.SuiteIDs = canonicalSuiteIDs(request.SuiteIDs)
	if len(request.SuiteIDs) == 0 {
		return request, fmt.Errorf("%w: builtin and installed suites cannot be mixed", ErrInvalid)
	}
	request.TrackIDs = canonicalTrackIDs(request.TrackIDs)
	if err := validateCapacityRunContract(
		request.Mode,
		request.TrackIDs,
		request.Concurrency,
		request.CapacitySLO,
		request.CapacityLoadProtocol,
	); err != nil {
		return request, err
	}
	if request.BaselineRunID != "" {
		if !validClientRequestID(request.BaselineRunID) {
			return request, fmt.Errorf("%w: invalid baseline_run_id", ErrInvalid)
		}
	}
	return request, nil
}

func executorTargetMatches(executor string, mode Mode, target CatalogTarget) bool {
	if !portableIDPattern.MatchString(executor) || !containsMode(target.Modes, mode) ||
		len(target.AcceptedExecutors) != len(target.Modes) {
		return false
	}
	for _, declaredMode := range target.Modes {
		executors, ok := target.AcceptedExecutors[declaredMode]
		if !ok || len(executors) == 0 {
			return false
		}
		seen := make(map[string]bool, len(executors))
		for _, accepted := range executors {
			if !portableIDPattern.MatchString(accepted) || seen[accepted] {
				return false
			}
			seen[accepted] = true
		}
	}
	for _, accepted := range target.AcceptedExecutors[mode] {
		if accepted == executor {
			return true
		}
	}
	return false
}

func validClientRequestID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}

func strictSuiteIDs(values []string) ([]string, error) {
	seen := make(map[string]bool, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		if value == "" || value != strings.TrimSpace(value) || seen[value] {
			return nil, fmt.Errorf("%w: suite_ids must be non-empty, trimmed, and unique", ErrInvalid)
		}
		seen[value] = true
		result = append(result, value)
	}
	return result, nil
}

func strictTrackIDs(values []TrackID) ([]TrackID, error) {
	seen := make(map[TrackID]bool, len(values))
	result := make([]TrackID, 0, len(values))
	for _, value := range values {
		if value == "" || string(value) != strings.TrimSpace(string(value)) || seen[value] {
			return nil, fmt.Errorf("%w: track_ids must be non-empty, trimmed, and unique", ErrInvalid)
		}
		seen[value] = true
		result = append(result, value)
	}
	return result, nil
}

func canonicalSuiteIDs(values []string) []string {
	selected := make(map[string]bool, len(values))
	for _, value := range values {
		selected[value] = true
	}
	builtin := make(map[string]bool, len(builtinSuites()))
	for _, suite := range builtinSuites() {
		builtin[suite.ID] = true
	}
	builtinCount := 0
	for value := range selected {
		if builtin[value] {
			builtinCount++
		}
	}
	if builtinCount != 0 && builtinCount != len(selected) {
		return nil
	}
	result := make([]string, 0, len(values))
	if builtinCount == 0 {
		for value := range selected {
			result = append(result, value)
		}
		sort.Strings(result)
		return result
	}
	for _, suite := range builtinSuites() {
		if selected[suite.ID] {
			result = append(result, suite.ID)
		}
	}
	return result
}

func canonicalTrackIDs(values []TrackID) []TrackID {
	selected := make(map[TrackID]bool, len(values))
	for _, value := range values {
		selected[value] = true
	}
	result := make([]TrackID, 0, len(values))
	for _, trackID := range allTrackIDs {
		if selected[trackID] {
			result = append(result, trackID)
		}
	}
	return result
}

func selectedSuiteEvidenceLevel(registry *Registry, suiteIDs []string, mode Mode) (EvidenceLevel, error) {
	selected := EvidenceLevel("")
	selectedRank := 6
	for _, suiteID := range suiteIDs {
		suite, ok := registry.suite(suiteID)
		if !ok {
			return "", fmt.Errorf("%w: unknown suite %q", ErrInvalid, suiteID)
		}
		executor, executable := suiteExecutorForMode(suite, mode)
		if !executable {
			return "", fmt.Errorf("%w: suite %q is not executable in mode %q", ErrInvalid, suiteID, mode)
		}
		contract, registered := registry.executor(executor)
		if !registered || contract.Mode != mode {
			return "", fmt.Errorf("%w: suite %q executor is not registered for mode %q", ErrInvalid, suiteID, mode)
		}
		rank := evidenceLevelRank(suite.EvidenceLevel)
		if rank < 0 {
			return "", fmt.Errorf("%w: suite %q has an invalid evidence level", ErrInvalid, suiteID)
		}
		level := suite.EvidenceLevel
		if contract.EvidenceLevelCeiling != "" {
			ceilingRank := evidenceLevelRank(contract.EvidenceLevelCeiling)
			if ceilingRank < rank {
				level, rank = contract.EvidenceLevelCeiling, ceilingRank
			}
		}
		if rank < selectedRank {
			selected, selectedRank = level, rank
		}
	}
	if selected == "" {
		return "", fmt.Errorf("%w: suite evidence level is unavailable", ErrInvalid)
	}
	return selected, nil
}

func evidenceLevelRank(level EvidenceLevel) int {
	switch level {
	case "E0":
		return 0
	case "E1":
		return 1
	case "E2":
		return 2
	case "E3":
		return 3
	case "E4":
		return 4
	case "E5":
		return 5
	default:
		return -1
	}
}
