package evaluationplane

import (
	"fmt"
	"math"
	"strings"
)

type gradingCaseEvidence struct {
	SchemaVersion  string   `json:"schema_version"`
	CaseID         string   `json:"case_id"`
	ExpectedRoute  *string  `json:"expected_route,omitempty"`
	ExpectedAnswer *string  `json:"expected_answer,omitempty"`
	PreferredArmID *string  `json:"preferred_arm_id,omitempty"`
	ExpectedTools  []string `json:"expected_tools"`
	ShouldBlock    *bool    `json:"should_block,omitempty"`
	Weight         float64  `json:"weight"`
}

func loadGradingCases(path string, caseIDs map[string]struct{}) (map[string]gradingCaseEvidence, error) {
	grading := make(map[string]gradingCaseEvidence, len(caseIDs))
	err := scanEvidenceJSONLines(path, maxWorkerArtifactBytes, maxCaseLineBytes, maxRecordsPerRun, func(line []byte, lineNumber int) error {
		var row gradingCaseEvidence
		if err := decodeStrictJSONLine(line, &row); err != nil {
			return fmt.Errorf("%w: grading-cases.jsonl line %d is invalid: %w", ErrInvalid, lineNumber, err)
		}
		if row.SchemaVersion != SchemaVersion || !evidenceIDPattern.MatchString(row.CaseID) ||
			!finiteFloat(row.Weight) || row.Weight <= 0 || row.ExpectedTools == nil {
			return fmt.Errorf("%w: grading-cases.jsonl line %d violates its contract", ErrInvalid, lineNumber)
		}
		if _, present := caseIDs[row.CaseID]; !present || grading[row.CaseID].CaseID != "" {
			return fmt.Errorf("%w: grading case identities do not match visible cases", ErrInvalid)
		}
		grading[row.CaseID] = row
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(grading) != len(caseIDs) {
		return nil, fmt.Errorf("%w: grading and visible case sets differ", ErrInvalid)
	}
	return grading, nil
}

func serverPoolOracleArmIDs(
	manifest RunManifest,
	entries []executionAttestationEntry,
	recordsByReceipt map[string][]executionRecordEvidence,
	cases visibleCaseSet,
	grading map[string]gradingCaseEvidence,
) map[string]map[string]struct{} {
	oracles := make(map[string]map[string]struct{})
	if manifest.Target.Mixture == nil || !containsTrack(manifest.TrackIDs, "routing") ||
		!containsTrack(manifest.TrackIDs, "model_pool") {
		return oracles
	}
	qualities := make(map[string]map[string]float64)
	for _, entry := range entries {
		if entry.Operation != workerBrokerArmChatCompletion {
			continue
		}
		bound := recordsByReceipt[entry.BrokerReceipt]
		if len(bound) != 1 {
			continue
		}
		record := bound[0]
		if record.TrackID != "model_pool" || record.ArmID == nil || entry.ArmID == nil ||
			record.CaseID != entry.CaseID || *record.ArmID != *entry.ArmID {
			continue
		}
		if _, routedCase := cases.CaseIDsByTrack["routing"][record.CaseID]; !routedCase {
			continue
		}
		quality := serverObservedAnswerQuality(entry, grading[record.CaseID])
		if quality == nil {
			continue
		}
		if qualities[record.CaseID] == nil {
			qualities[record.CaseID] = make(map[string]float64, len(manifest.Target.Mixture.ModelArms))
		}
		qualities[record.CaseID][*entry.ArmID] = *quality
	}
	for caseID := range cases.CaseIDsByTrack["routing"] {
		if _, poolCase := cases.CaseIDsByTrack["model_pool"][caseID]; !poolCase ||
			len(qualities[caseID]) != len(manifest.Target.Mixture.ModelArms) {
			continue
		}
		complete := true
		maximum := -1.0
		for _, arm := range manifest.Target.Mixture.ModelArms {
			quality, present := qualities[caseID][arm.ID]
			if !present {
				complete = false
				break
			}
			if quality > maximum {
				maximum = quality
			}
		}
		if !complete {
			continue
		}
		oracle := make(map[string]struct{})
		for _, arm := range manifest.Target.Mixture.ModelArms {
			if qualities[caseID][arm.ID] == maximum {
				oracle[arm.ID] = struct{}{}
			}
		}
		oracles[caseID] = oracle
	}
	return oracles
}

func serverObservedQuality(
	entry executionAttestationEntry,
	trackID TrackID,
	grading gradingCaseEvidence,
	poolOracleArmIDs map[string]struct{},
) *float64 {
	if !entry.Success {
		return nil
	}
	switch trackID {
	case "routing":
		if entry.ArmID == nil {
			return nil
		}
		value := 0.0
		if grading.ExpectedRoute != nil {
			if *grading.ExpectedRoute == *entry.ArmID ||
				(entry.SelectedModel != nil && *grading.ExpectedRoute == *entry.SelectedModel) {
				value = 1
			}
			return &value
		}
		if len(poolOracleArmIDs) == 0 {
			return nil
		}
		if _, oracle := poolOracleArmIDs[*entry.ArmID]; oracle {
			value = 1
		}
		return &value
	case "model_pool", "joint", "multimodal":
		return serverObservedAnswerQuality(entry, grading)
	default:
		return nil
	}
}

func serverObservedAnswerQuality(entry executionAttestationEntry, grading gradingCaseEvidence) *float64 {
	if !entry.Success || grading.ExpectedAnswer == nil || entry.ResponseContentDigest == nil {
		return nil
	}
	value := 0.0
	if digestString(normalizedAnswer(*grading.ExpectedAnswer)) == *entry.ResponseContentDigest {
		value = 1
	}
	return &value
}

func brokerResponseContent(payload map[string]any) *string {
	choices, ok := payload["choices"].([]any)
	if !ok || len(choices) == 0 {
		return nil
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		return nil
	}
	message, ok := choice["message"].(map[string]any)
	if !ok {
		return nil
	}
	// Reasoning models can exhaust their output budget before emitting a final
	// content field. Bind an available reasoning field as the exact observed
	// provider output; grading still uses normalized exact-text comparison.
	for _, field := range []string{"content", "reasoning", "reasoning_content"} {
		content, present := message[field].(string)
		if present {
			return &content
		}
	}
	return nil
}

func normalizedAnswer(value string) string {
	return strings.Join(strings.Fields(value), " ")
}

func serverRuntimeCost(entry executionAttestationEntry, arms []ModelArm) *float64 {
	if entry.ArmID == nil || entry.InputTokens == nil || entry.OutputTokens == nil {
		return nil
	}
	for _, arm := range arms {
		if *entry.ArmID != arm.ID {
			continue
		}
		value := (float64(*entry.InputTokens)*arm.InputCostPerMillionTokensUSD +
			float64(*entry.OutputTokens)*arm.OutputCostPerMillionTokensUSD) / 1_000_000
		return &value
	}
	return nil
}

func sameOptionalString(left, right *string) bool {
	return (left == nil && right == nil) || (left != nil && right != nil && *left == *right)
}

func sameOptionalFloat(left, right *float64) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return !math.IsNaN(*left) && !math.IsNaN(*right) && *left == *right
}
