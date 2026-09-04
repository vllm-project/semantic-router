package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

type visibleCaseSet struct {
	IDs            map[string]struct{}
	Modalities     map[string]string
	TrackIDsByCase map[string][]TrackID
	CaseIDsByTrack map[TrackID]map[string]struct{}
	MessageDigests map[string]string
}

type visibleCaseIdentity struct {
	SchemaVersion string           `json:"schema_version"`
	ID            string           `json:"id"`
	TrackIDs      []TrackID        `json:"track_ids"`
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

func manifestVisibleCaseLimit(manifest RunManifest, executor executorContract) (int, error) {
	if !executor.CaseBudgetPerSuite {
		return manifest.SampleLimit, nil
	}
	if manifest.SampleLimit < 1 || len(manifest.SuiteIDs) == 0 || len(manifest.SuiteIDs) > maxRecordsPerRun/manifest.SampleLimit {
		return 0, fmt.Errorf("%w: normalized suite case budget is invalid", ErrInvalid)
	}
	return manifest.SampleLimit * len(manifest.SuiteIDs), nil
}

func validateVisibleCaseSet(path string, caseLimit int, selectedTrackIDs []TrackID) (visibleCaseSet, error) {
	if caseLimit < 1 || caseLimit > maxRecordsPerRun {
		return visibleCaseSet{}, fmt.Errorf("%w: run manifest case limit is invalid", ErrInvalid)
	}
	if len(selectedTrackIDs) == 0 || !canonicalTrackOrder(selectedTrackIDs) {
		return visibleCaseSet{}, fmt.Errorf("%w: run manifest track plan is invalid", ErrInvalid)
	}
	selected := make(map[TrackID]struct{}, len(selectedTrackIDs))
	for _, trackID := range selectedTrackIDs {
		selected[trackID] = struct{}{}
	}
	cases := visibleCaseSet{
		IDs: make(map[string]struct{}), Modalities: make(map[string]string),
		TrackIDsByCase: make(map[string][]TrackID), CaseIDsByTrack: make(map[TrackID]map[string]struct{}, len(selectedTrackIDs)),
		MessageDigests: make(map[string]string),
	}
	for _, trackID := range selectedTrackIDs {
		cases.CaseIDsByTrack[trackID] = make(map[string]struct{})
	}
	err := scanEvidenceJSONLines(path, maxWorkerArtifactBytes, maxCaseLineBytes, caseLimit, func(line []byte, lineNumber int) error {
		var identity visibleCaseIdentity
		if err := decodeStrictJSONLine(line, &identity); err != nil {
			return fmt.Errorf("%w: cases.jsonl line %d is invalid: %w", ErrInvalid, lineNumber, err)
		}
		if identity.SchemaVersion != SchemaVersion || !evidenceIDPattern.MatchString(identity.ID) || len(identity.TrackIDs) == 0 ||
			!canonicalTrackOrder(identity.TrackIDs) || len(identity.Messages) == 0 || !validCaseModality(identity.Modality) {
			return fmt.Errorf("%w: cases.jsonl line %d violates the visible case contract", ErrInvalid, lineNumber)
		}
		for _, trackID := range identity.TrackIDs {
			if _, planned := selected[trackID]; !planned {
				return fmt.Errorf("%w: cases.jsonl line %d plans unselected track %q", ErrInvalid, lineNumber, trackID)
			}
			if identity.Modality == "text" && trackID == "multimodal" {
				return fmt.Errorf("%w: cases.jsonl line %d plans multimodal evidence for a text case", ErrInvalid, lineNumber)
			}
		}
		for index, message := range identity.Messages {
			if err := validateVisibleMessage(message); err != nil {
				return fmt.Errorf("%w: cases.jsonl line %d message %d is invalid: %w", ErrInvalid, lineNumber, index+1, err)
			}
		}
		messagesDigest, err := canonicalMessageListDigest(identity.Messages)
		if err != nil {
			return fmt.Errorf("%w: cases.jsonl line %d messages cannot be digested: %w", ErrInvalid, lineNumber, err)
		}
		if _, duplicate := cases.IDs[identity.ID]; duplicate {
			return fmt.Errorf("%w: cases.jsonl contains duplicate case id %q", ErrInvalid, identity.ID)
		}
		cases.IDs[identity.ID] = struct{}{}
		cases.Modalities[identity.ID] = identity.Modality
		cases.TrackIDsByCase[identity.ID] = append([]TrackID(nil), identity.TrackIDs...)
		cases.MessageDigests[identity.ID] = messagesDigest
		for _, trackID := range identity.TrackIDs {
			cases.CaseIDsByTrack[trackID][identity.ID] = struct{}{}
		}
		return nil
	})
	if err != nil {
		return visibleCaseSet{}, err
	}
	for _, trackID := range selectedTrackIDs {
		if len(cases.CaseIDsByTrack[trackID]) == 0 {
			return visibleCaseSet{}, fmt.Errorf("%w: selected track %q has no planned visible case", ErrInvalid, trackID)
		}
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
