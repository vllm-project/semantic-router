package evaluationplane

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	maxCaseLineBytes   = 16 * 1024 * 1024
	maxRecordLineBytes = 256 * 1024
	maxRecordsPerRun   = 1_000_000
)

var evidenceIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)

type recordStatusCounts struct {
	Succeeded   int
	Failed      int
	Unavailable int
}

func (counts recordStatusCounts) total() int {
	return counts.Succeeded + counts.Failed + counts.Unavailable
}

type recordAttestation struct {
	validated   bool
	Total       int
	Succeeded   int
	Failed      int
	Unavailable int
	ByTrack     map[TrackID]recordStatusCounts
}

func (attestation recordAttestation) validatesGateCoverage(gate Gate) bool {
	if !attestation.validated || attestation.Total < 1 || gate.SampleCount == nil || gate.Coverage == nil {
		return false
	}
	evaluated := attestation.Succeeded + attestation.Failed
	expectedFraction := float64(evaluated) / float64(attestation.Total)
	return *gate.SampleCount == evaluated &&
		gate.Coverage.Evaluated == evaluated &&
		gate.Coverage.Total == attestation.Total &&
		gate.Coverage.Unavailable == attestation.Unavailable &&
		finiteFloat(gate.Coverage.Fraction) && gate.Coverage.Fraction == expectedFraction
}

type visibleCaseIdentity struct {
	SchemaVersion string           `json:"schema_version"`
	ID            string           `json:"id"`
	Messages      []visibleMessage `json:"messages"`
	Modality      string           `json:"modality"`
	Tags          []string         `json:"tags"`
	TrajectoryID  *string          `json:"trajectory_id,omitempty"`
}

type visibleMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	Name       *string         `json:"name,omitempty"`
	ToolCallID *string         `json:"tool_call_id,omitempty"`
}

type textContentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type imageURL struct {
	URL    string  `json:"url"`
	Detail *string `json:"detail,omitempty"`
}

type imageContentPart struct {
	Type     string   `json:"type"`
	ImageURL imageURL `json:"image_url"`
}

type executionRecordEvidence struct {
	SchemaVersion      string   `json:"schema_version"`
	ID                 string   `json:"id"`
	TrackID            TrackID  `json:"track_id"`
	CaseID             string   `json:"case_id"`
	AttemptID          string   `json:"attempt_id"`
	Status             string   `json:"status"`
	ArmID              *string  `json:"arm_id,omitempty"`
	SelectedArmID      *string  `json:"selected_arm_id,omitempty"`
	SelectionStatus    *string  `json:"selection_status,omitempty"`
	SelectionMethod    *string  `json:"selection_method,omitempty"`
	Recipe             *string  `json:"recipe,omitempty"`
	DecisionName       *string  `json:"decision_name,omitempty"`
	Algorithm          *string  `json:"algorithm,omitempty"`
	TraceDigest        *string  `json:"trace_digest,omitempty"`
	Success            *bool    `json:"success,omitempty"`
	Quality            *float64 `json:"quality,omitempty"`
	Fallback           *bool    `json:"fallback,omitempty"`
	LatencyMS          *float64 `json:"latency_ms,omitempty"`
	InputTokens        *int64   `json:"input_tokens,omitempty"`
	OutputTokens       *int64   `json:"output_tokens,omitempty"`
	RuntimeCost        *float64 `json:"runtime_cost,omitempty"`
	EvaluationCost     *float64 `json:"evaluation_cost,omitempty"`
	CapacityTCO        *float64 `json:"capacity_tco,omitempty"`
	TrajectorySteps    *int64   `json:"trajectory_steps,omitempty"`
	ToolCalls          *int64   `json:"tool_calls,omitempty"`
	InvalidToolCalls   *int64   `json:"invalid_tool_calls,omitempty"`
	Modality           *string  `json:"modality,omitempty"`
	PrivacyViolations  *int64   `json:"privacy_violations,omitempty"`
	PreferenceMatch    *bool    `json:"preference_match,omitempty"`
	BehaviorPropensity *float64 `json:"behavior_propensity,omitempty"`
	SafetyViolations   *int64   `json:"safety_violations,omitempty"`
	ShouldBlock        *bool    `json:"should_block,omitempty"`
	Blocked            *bool    `json:"blocked,omitempty"`
	Concurrency        *int64   `json:"concurrency,omitempty"`
	ThroughputRPS      *float64 `json:"throughput_rps,omitempty"`
	GPUSeconds         *float64 `json:"gpu_seconds,omitempty"`
	EnergyKWh          *float64 `json:"energy_kwh,omitempty"`
	LoadElapsedSeconds *float64 `json:"load_elapsed_seconds,omitempty"`
	Grader             *string  `json:"grader,omitempty"`
	EvidenceKind       *string  `json:"evidence_kind,omitempty"`
	Error              *string  `json:"error,omitempty"`
}

type failureTrackSummary struct {
	TrackID     TrackID `json:"track_id"`
	Succeeded   int     `json:"succeeded"`
	Failed      int     `json:"failed"`
	Unavailable int     `json:"unavailable"`
}

type failureSummaryEvidence struct {
	SchemaVersion string                `json:"schema_version"`
	TotalRecords  int                   `json:"total_records"`
	Failed        int                   `json:"failed"`
	Unavailable   int                   `json:"unavailable"`
	ByTrack       []failureTrackSummary `json:"by_track"`
}

type recordSemanticKey struct {
	TrackID       TrackID
	CaseID        string
	AttemptID     string
	ArmID         string
	SelectedArmID string
}

func validateRecordsAndFailureSummary(runDir string, manifest RunManifest) (recordAttestation, error) {
	caseIDs, err := validateVisibleCases(filepath.Join(runDir, "cases.jsonl"), manifest.SampleLimit)
	if err != nil {
		return recordAttestation{}, err
	}
	attestation, err := validateExecutionRecords(filepath.Join(runDir, "records.jsonl"), manifest.TrackIDs, caseIDs)
	if err != nil {
		return recordAttestation{}, err
	}
	if err := validateFailureSummaryAgainstRecords(filepath.Join(runDir, "failure-summary.json"), attestation); err != nil {
		return recordAttestation{}, err
	}
	attestation.validated = true
	return attestation, nil
}

func validateVisibleCases(path string, sampleLimit int) (map[string]struct{}, error) {
	if sampleLimit < 1 || sampleLimit > maxSampleLimit {
		return nil, fmt.Errorf("%w: run manifest sample limit is invalid", ErrInvalid)
	}
	cases := make(map[string]struct{})
	err := scanEvidenceJSONLines(path, maxWorkerArtifactBytes, maxCaseLineBytes, sampleLimit, func(line []byte, lineNumber int) error {
		var identity visibleCaseIdentity
		if err := decodeStrictJSONLine(line, &identity); err != nil {
			return fmt.Errorf("%w: cases.jsonl line %d is invalid: %w", ErrInvalid, lineNumber, err)
		}
		if identity.SchemaVersion != SchemaVersion || !evidenceIDPattern.MatchString(identity.ID) || len(identity.Messages) == 0 || !validCaseModality(identity.Modality) {
			return fmt.Errorf("%w: cases.jsonl line %d violates the visible case contract", ErrInvalid, lineNumber)
		}
		for index, message := range identity.Messages {
			if err := validateVisibleMessage(message); err != nil {
				return fmt.Errorf("%w: cases.jsonl line %d message %d is invalid: %w", ErrInvalid, lineNumber, index+1, err)
			}
		}
		if _, duplicate := cases[identity.ID]; duplicate {
			return fmt.Errorf("%w: cases.jsonl contains duplicate case id %q", ErrInvalid, identity.ID)
		}
		cases[identity.ID] = struct{}{}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return cases, nil
}

func validateVisibleMessage(message visibleMessage) error {
	switch message.Role {
	case "system", "user", "assistant", "tool":
	default:
		return fmt.Errorf("role %q is unsupported", message.Role)
	}
	if len(message.Content) == 0 || bytes.Equal(bytes.TrimSpace(message.Content), []byte("null")) {
		return fmt.Errorf("content is required")
	}
	var text string
	if err := json.Unmarshal(message.Content, &text); err == nil {
		if strings.TrimSpace(text) == "" {
			return fmt.Errorf("string content must be non-empty")
		}
		return nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(message.Content, &parts); err != nil {
		return fmt.Errorf("content must be a string or typed content-part array")
	}
	if len(parts) == 0 {
		return fmt.Errorf("content-part array must be non-empty")
	}
	for index, raw := range parts {
		var header struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(raw, &header); err != nil {
			return fmt.Errorf("content part %d is not an object", index+1)
		}
		switch header.Type {
		case "text":
			var part textContentPart
			if err := decodeStrictJSONLine(raw, &part); err != nil || part.Type != "text" || strings.TrimSpace(part.Text) == "" {
				return fmt.Errorf("content part %d violates the text contract", index+1)
			}
		case "image_url":
			var part imageContentPart
			if err := decodeStrictJSONLine(raw, &part); err != nil || part.Type != "image_url" || strings.TrimSpace(part.ImageURL.URL) == "" {
				return fmt.Errorf("content part %d violates the image_url contract", index+1)
			}
			if part.ImageURL.Detail != nil && *part.ImageURL.Detail != "auto" && *part.ImageURL.Detail != "low" && *part.ImageURL.Detail != "high" {
				return fmt.Errorf("content part %d has an invalid image detail", index+1)
			}
		default:
			return fmt.Errorf("content part %d has unsupported type %q", index+1, header.Type)
		}
	}
	return nil
}

func validCaseModality(modality string) bool {
	switch modality {
	case "text", "image", "document", "audio", "video":
		return true
	default:
		return false
	}
}

func validateExecutionRecords(path string, selectedTracks []TrackID, caseIDs map[string]struct{}) (recordAttestation, error) {
	selected := make(map[TrackID]bool, len(selectedTracks))
	for _, trackID := range selectedTracks {
		selected[trackID] = true
	}
	attestation := recordAttestation{ByTrack: make(map[TrackID]recordStatusCounts, len(selectedTracks))}
	recordIDs := make(map[string]struct{})
	semanticKeys := make(map[recordSemanticKey]string)
	err := scanEvidenceJSONLines(path, maxWorkerArtifactBytes, maxRecordLineBytes, maxRecordsPerRun, func(line []byte, lineNumber int) error {
		var record executionRecordEvidence
		if err := decodeStrictJSONLine(line, &record); err != nil {
			return fmt.Errorf("%w: records.jsonl line %d is invalid: %w", ErrInvalid, lineNumber, err)
		}
		if err := validateExecutionRecord(record, selected, caseIDs); err != nil {
			return fmt.Errorf("%w: records.jsonl line %d: %w", ErrInvalid, lineNumber, err)
		}
		if _, duplicate := recordIDs[record.ID]; duplicate {
			return fmt.Errorf("%w: records.jsonl contains duplicate record id %q", ErrInvalid, record.ID)
		}
		semanticKey := record.semanticKey()
		if priorID, duplicate := semanticKeys[semanticKey]; duplicate {
			return fmt.Errorf("%w: records.jsonl record %q duplicates semantic attempt %q", ErrInvalid, record.ID, priorID)
		}
		recordIDs[record.ID] = struct{}{}
		semanticKeys[semanticKey] = record.ID
		counts := attestation.ByTrack[record.TrackID]
		switch record.Status {
		case "succeeded":
			counts.Succeeded++
			attestation.Succeeded++
		case "failed":
			counts.Failed++
			attestation.Failed++
		case "unavailable":
			counts.Unavailable++
			attestation.Unavailable++
		}
		attestation.ByTrack[record.TrackID] = counts
		attestation.Total++
		return nil
	})
	if err != nil {
		return recordAttestation{}, err
	}
	for _, trackID := range selectedTracks {
		if attestation.ByTrack[trackID].total() == 0 {
			return recordAttestation{}, fmt.Errorf("%w: records.jsonl has no evidence for selected track %q", ErrInvalid, trackID)
		}
	}
	return attestation, nil
}

func (record executionRecordEvidence) semanticKey() recordSemanticKey {
	key := recordSemanticKey{TrackID: record.TrackID, CaseID: record.CaseID, AttemptID: record.AttemptID}
	if record.ArmID != nil {
		key.ArmID = *record.ArmID
	}
	if record.SelectedArmID != nil {
		key.SelectedArmID = *record.SelectedArmID
	}
	return key
}

func validateExecutionRecord(record executionRecordEvidence, selectedTracks map[TrackID]bool, caseIDs map[string]struct{}) error {
	if record.SchemaVersion != SchemaVersion {
		return fmt.Errorf("schema_version must be %q", SchemaVersion)
	}
	if !evidenceIDPattern.MatchString(record.ID) || !evidenceIDPattern.MatchString(record.CaseID) || !evidenceIDPattern.MatchString(record.AttemptID) {
		return fmt.Errorf("record id, case_id, and attempt_id must be portable non-empty identities")
	}
	if !selectedTracks[record.TrackID] {
		return fmt.Errorf("track_id %q is not selected by the immutable manifest", record.TrackID)
	}
	if _, ok := caseIDs[record.CaseID]; !ok {
		return fmt.Errorf("case_id %q is absent from the validated case set", record.CaseID)
	}
	if record.Status != "succeeded" && record.Status != "failed" && record.Status != "unavailable" {
		return fmt.Errorf("status %q is invalid", record.Status)
	}
	if record.TraceDigest != nil && !digestPattern.MatchString(*record.TraceDigest) {
		return fmt.Errorf("trace_digest is invalid")
	}
	for _, identity := range []*string{record.ArmID, record.SelectedArmID} {
		if identity != nil && (strings.TrimSpace(*identity) == "" || len(*identity) > 512) {
			return fmt.Errorf("arm identities must be non-empty and bounded")
		}
	}
	if err := validateRecordNumbers(record); err != nil {
		return err
	}
	return nil
}

func validateRecordNumbers(record executionRecordEvidence) error {
	for _, field := range []struct {
		name  string
		value *float64
	}{
		{"latency_ms", record.LatencyMS},
		{"runtime_cost", record.RuntimeCost},
		{"evaluation_cost", record.EvaluationCost},
		{"capacity_tco", record.CapacityTCO},
		{"throughput_rps", record.ThroughputRPS},
		{"gpu_seconds", record.GPUSeconds},
		{"energy_kwh", record.EnergyKWh},
		{"load_elapsed_seconds", record.LoadElapsedSeconds},
	} {
		if field.value != nil && (!finiteFloat(*field.value) || *field.value < 0) {
			return fmt.Errorf("%s must be finite and non-negative", field.name)
		}
	}
	if record.Quality != nil && (!finiteFloat(*record.Quality) || *record.Quality < 0 || *record.Quality > 1) {
		return fmt.Errorf("quality must be a finite fraction")
	}
	if record.BehaviorPropensity != nil && (!finiteFloat(*record.BehaviorPropensity) || *record.BehaviorPropensity <= 0 || *record.BehaviorPropensity > 1) {
		return fmt.Errorf("behavior_propensity must be a finite positive fraction")
	}
	for _, field := range []struct {
		name  string
		value *int64
	}{
		{"input_tokens", record.InputTokens},
		{"output_tokens", record.OutputTokens},
		{"trajectory_steps", record.TrajectorySteps},
		{"tool_calls", record.ToolCalls},
		{"invalid_tool_calls", record.InvalidToolCalls},
		{"privacy_violations", record.PrivacyViolations},
		{"safety_violations", record.SafetyViolations},
	} {
		if field.value != nil && *field.value < 0 {
			return fmt.Errorf("%s must be non-negative", field.name)
		}
	}
	if record.Concurrency != nil && *record.Concurrency < 1 {
		return fmt.Errorf("concurrency must be positive")
	}
	return nil
}

func validateFailureSummaryAgainstRecords(path string, attestation recordAttestation) error {
	var summary failureSummaryEvidence
	if err := decodeStrictEvidence(path, &summary); err != nil {
		return err
	}
	if summary.SchemaVersion != SchemaVersion || summary.TotalRecords != attestation.Total ||
		summary.Failed != attestation.Failed || summary.Unavailable != attestation.Unavailable {
		return fmt.Errorf("%w: failure-summary.json does not match validated records", ErrInvalid)
	}
	trackIDs := make([]string, 0, len(attestation.ByTrack))
	for trackID := range attestation.ByTrack {
		trackIDs = append(trackIDs, string(trackID))
	}
	sort.Strings(trackIDs)
	if len(summary.ByTrack) != len(trackIDs) {
		return fmt.Errorf("%w: failure-summary.json track set does not match validated records", ErrInvalid)
	}
	for index, trackID := range trackIDs {
		counts := attestation.ByTrack[TrackID(trackID)]
		actual := summary.ByTrack[index]
		if actual.TrackID != TrackID(trackID) || actual.Succeeded != counts.Succeeded ||
			actual.Failed != counts.Failed || actual.Unavailable != counts.Unavailable {
			return fmt.Errorf("%w: failure-summary.json track counts do not match validated records", ErrInvalid)
		}
	}
	return nil
}

func scanEvidenceJSONLines(path string, maxBytes int64, maxLineBytes, maxLines int, visit func([]byte, int) error) error {
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		return fmt.Errorf("open evidence file %s: %w", filepath.Base(path), err)
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("stat evidence file %s: %w", filepath.Base(path), err)
	}
	if info.Size() < 1 || info.Size() > maxBytes {
		return fmt.Errorf("%w: evidence file %s violates its total-byte limit", ErrInvalid, filepath.Base(path))
	}
	if _, err := file.Seek(-1, io.SeekEnd); err != nil {
		return fmt.Errorf("inspect evidence file %s: %w", filepath.Base(path), err)
	}
	var ending [1]byte
	if _, err := io.ReadFull(file, ending[:]); err != nil || ending[0] != '\n' {
		return fmt.Errorf("%w: evidence file %s must end with a newline", ErrInvalid, filepath.Base(path))
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("rewind evidence file %s: %w", filepath.Base(path), err)
	}
	initialBuffer := maxLineBytes + 1
	if initialBuffer > 64*1024 {
		initialBuffer = 64 * 1024
	}
	scanner := bufio.NewScanner(io.LimitReader(file, maxBytes+1))
	scanner.Buffer(make([]byte, initialBuffer), maxLineBytes+1)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := scanner.Bytes()
		if lineNumber > maxLines {
			return fmt.Errorf("%w: evidence file %s exceeds its line-count limit", ErrInvalid, filepath.Base(path))
		}
		if len(line) == 0 || len(line) > maxLineBytes {
			return fmt.Errorf("%w: evidence file %s line %d violates its line-byte limit", ErrInvalid, filepath.Base(path), lineNumber)
		}
		if err := visit(line, lineNumber); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("%w: scan evidence file %s: %w", ErrInvalid, filepath.Base(path), err)
	}
	if lineNumber == 0 {
		return fmt.Errorf("%w: evidence file %s must contain at least one row", ErrInvalid, filepath.Base(path))
	}
	return nil
}

func decodeStrictJSONLine(line []byte, destination any) error {
	decoder := json.NewDecoder(bytes.NewReader(line))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	return ensureJSONEOF(decoder)
}
