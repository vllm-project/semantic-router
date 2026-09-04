package evaluationplane

import (
	"fmt"
	"strings"
)

const (
	maxRunNameLength        = 200
	maxRunDescriptionLength = 4000
	maxSampleLimit          = 100000
	maxRunConcurrency       = 128
)

func (s *Service) validateCreateRequest(registry *Registry, request CreateRunRequest) (CreateRunRequest, targetDefinition, error) {
	request.Name = strings.TrimSpace(request.Name)
	request.Description = strings.TrimSpace(request.Description)
	request.TargetID = strings.TrimSpace(request.TargetID)
	request.ChangeProfile = ChangeProfile(strings.TrimSpace(string(request.ChangeProfile)))
	request.BaselineRunID = strings.TrimSpace(request.BaselineRunID)
	request.SuiteIDs = uniqueStrings(request.SuiteIDs)
	request.TrackIDs = uniqueTracks(request.TrackIDs)
	if request.Name == "" || len(request.Name) > maxRunNameLength {
		return request, targetDefinition{}, fmt.Errorf("%w: name must be 1-%d characters", ErrInvalid, maxRunNameLength)
	}
	if len(request.Description) > maxRunDescriptionLength {
		return request, targetDefinition{}, fmt.Errorf("%w: description exceeds %d characters", ErrInvalid, maxRunDescriptionLength)
	}
	if request.Mode != ModeReplay && request.Mode != ModeLive {
		return request, targetDefinition{}, fmt.Errorf("%w: mode must be replay or live", ErrInvalid)
	}
	if request.SampleLimit < 1 || request.SampleLimit > maxSampleLimit {
		return request, targetDefinition{}, fmt.Errorf("%w: sample_limit must be between 1 and %d", ErrInvalid, maxSampleLimit)
	}
	if request.Concurrency < 1 || request.Concurrency > maxRunConcurrency {
		return request, targetDefinition{}, fmt.Errorf("%w: concurrency must be between 1 and %d", ErrInvalid, maxRunConcurrency)
	}
	if request.Seed < 0 || request.Seed > 1<<32-1 {
		return request, targetDefinition{}, fmt.Errorf("%w: seed must be between 0 and %d", ErrInvalid, uint64(1<<32-1))
	}
	if request.AutoStart {
		return request, targetDefinition{}, fmt.Errorf("%w: auto_start must be false; use the separately authorized start endpoint", ErrInvalid)
	}
	if len(request.SuiteIDs) == 0 || len(request.TrackIDs) == 0 {
		return request, targetDefinition{}, fmt.Errorf("%w: at least one suite and track are required", ErrInvalid)
	}
	if _, ok := registry.changeProfile(request.ChangeProfile); !ok {
		return request, targetDefinition{}, fmt.Errorf("%w: unknown change_profile", ErrInvalid)
	}
	target, ok := registry.target(request.TargetID)
	if !ok {
		return request, targetDefinition{}, fmt.Errorf("%w: unknown target_id", ErrInvalid)
	}
	if !containsMode(target.Public.Modes, request.Mode) {
		return request, targetDefinition{}, fmt.Errorf("%w: target does not support requested mode", ErrInvalid)
	}
	if request.Mode == ModeLive && !digestPattern.MatchString(target.BackendTopologyDigest) {
		return request, targetDefinition{}, fmt.Errorf("%w: live target backend topology identity is unavailable", ErrInvalid)
	}
	selectedSuiteTracks := make(map[TrackID]bool)
	for _, suiteID := range request.SuiteIDs {
		suite, ok := registry.suite(suiteID)
		if !ok {
			return request, targetDefinition{}, fmt.Errorf("%w: unknown suite %q", ErrInvalid, suiteID)
		}
		if !containsMode(suite.Modes, request.Mode) {
			return request, targetDefinition{}, fmt.Errorf("%w: suite %q does not support requested mode", ErrInvalid, suiteID)
		}
		for _, trackID := range suite.TrackIDs {
			selectedSuiteTracks[trackID] = true
			if !containsTrack(target.Public.TrackIDs, trackID) {
				return request, targetDefinition{}, fmt.Errorf("%w: target cannot execute every track required by suite %q", ErrInvalid, suiteID)
			}
		}
	}
	for _, trackID := range request.TrackIDs {
		track, ok := registry.track(trackID)
		if !ok {
			return request, targetDefinition{}, fmt.Errorf("%w: unknown track %q", ErrInvalid, trackID)
		}
		if !selectedSuiteTracks[trackID] {
			return request, targetDefinition{}, fmt.Errorf("%w: track %q is not provided by selected suites", ErrInvalid, trackID)
		}
		if !containsMode(track.Modes, request.Mode) || !containsTrack(target.Public.TrackIDs, trackID) {
			return request, targetDefinition{}, fmt.Errorf("%w: target cannot execute track %q in requested mode", ErrInvalid, trackID)
		}
	}
	if request.BaselineRunID != "" {
		if err := validateResourceID(request.BaselineRunID); err != nil {
			return request, targetDefinition{}, fmt.Errorf("%w: invalid baseline_run_id", ErrInvalid)
		}
	}
	return request, target, nil
}

func uniqueStrings(values []string) []string {
	seen := make(map[string]bool, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" && !seen[value] {
			seen[value] = true
			result = append(result, value)
		}
	}
	return result
}

func uniqueTracks(values []TrackID) []TrackID {
	seen := make(map[TrackID]bool, len(values))
	result := make([]TrackID, 0, len(values))
	for _, value := range values {
		value = TrackID(strings.TrimSpace(string(value)))
		if value != "" && !seen[value] {
			seen[value] = true
			result = append(result, value)
		}
	}
	return result
}

func selectedSuiteEvidenceLevel(registry *Registry, suiteIDs []string) (EvidenceLevel, error) {
	selected := EvidenceLevel("")
	selectedRank := 6
	for _, suiteID := range suiteIDs {
		suite, ok := registry.suite(suiteID)
		if !ok {
			return "", fmt.Errorf("%w: unknown suite %q", ErrInvalid, suiteID)
		}
		rank := evidenceLevelRank(suite.EvidenceLevel)
		if rank < 0 {
			return "", fmt.Errorf("%w: suite %q has an invalid evidence level", ErrInvalid, suiteID)
		}
		if rank < selectedRank {
			selected, selectedRank = suite.EvidenceLevel, rank
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
