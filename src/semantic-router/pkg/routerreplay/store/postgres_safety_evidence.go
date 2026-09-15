package store

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// New safety metadata lives in one additive JSONB column. Historical rows
// without it keep score availability false rather than guessing from zero.
type postgresSafetyEvidence struct {
	ConfidenceScoreAvailable        bool                 `json:"confidence_score_available"`
	SignalErrorMatches              map[string]bool      `json:"signal_error_matches,omitempty"`
	JailbreakDetected               bool                 `json:"jailbreak_detected,omitempty"`
	JailbreakType                   string               `json:"jailbreak_type,omitempty"`
	JailbreakConfidence             *float32             `json:"jailbreak_confidence,omitempty"`
	JailbreakScoreAvailable         bool                 `json:"jailbreak_score_available"`
	JailbreakDecision               *tasks.LabelDecision `json:"jailbreak_decision,omitempty"`
	ResponseJailbreakDetected       bool                 `json:"response_jailbreak_detected,omitempty"`
	ResponseJailbreakType           string               `json:"response_jailbreak_type,omitempty"`
	ResponseJailbreakConfidence     *float32             `json:"response_jailbreak_confidence,omitempty"`
	ResponseJailbreakScoreAvailable bool                 `json:"response_jailbreak_score_available"`
	ResponseJailbreakDecision       *tasks.LabelDecision `json:"response_jailbreak_decision,omitempty"`
	HallucinationScoreAvailable     bool                 `json:"hallucination_score_available"`
	HallucinationScoreKind          string               `json:"hallucination_score_kind,omitempty"`
}

func marshalPostgresSafety(record Record) ([]byte, error) {
	return json.Marshal(postgresSafetyEvidence{
		ConfidenceScoreAvailable: record.ConfidenceScoreAvailable, SignalErrorMatches: record.SignalErrorMatches,
		JailbreakDetected: record.JailbreakDetected, JailbreakType: record.JailbreakType,
		JailbreakConfidence:     availableFloat32(record.JailbreakConfidence, record.JailbreakScoreAvailable),
		JailbreakScoreAvailable: record.JailbreakScoreAvailable, JailbreakDecision: record.JailbreakDecision,
		ResponseJailbreakDetected: record.ResponseJailbreakDetected, ResponseJailbreakType: record.ResponseJailbreakType,
		ResponseJailbreakConfidence:     availableFloat32(record.ResponseJailbreakConfidence, record.ResponseJailbreakScoreAvailable),
		ResponseJailbreakScoreAvailable: record.ResponseJailbreakScoreAvailable, ResponseJailbreakDecision: record.ResponseJailbreakDecision,
		HallucinationScoreAvailable: record.HallucinationScoreAvailable, HallucinationScoreKind: record.HallucinationScoreKind,
	})
}

func unmarshalPostgresSafety(encoded []byte, record *Record) error {
	var evidence postgresSafetyEvidence
	if err := unmarshalReplayOptionalJSON(encoded, &evidence); err != nil {
		return err
	}
	record.ConfidenceScoreAvailable = evidence.ConfidenceScoreAvailable
	record.SignalErrorMatches = evidence.SignalErrorMatches
	record.JailbreakDetected = evidence.JailbreakDetected
	record.JailbreakType = evidence.JailbreakType
	if evidence.JailbreakConfidence != nil {
		record.JailbreakConfidence = *evidence.JailbreakConfidence
	}
	record.JailbreakScoreAvailable = evidence.JailbreakScoreAvailable
	record.JailbreakDecision = evidence.JailbreakDecision
	record.ResponseJailbreakDetected = evidence.ResponseJailbreakDetected
	record.ResponseJailbreakType = evidence.ResponseJailbreakType
	if evidence.ResponseJailbreakConfidence != nil {
		record.ResponseJailbreakConfidence = *evidence.ResponseJailbreakConfidence
	}
	record.ResponseJailbreakScoreAvailable = evidence.ResponseJailbreakScoreAvailable
	record.ResponseJailbreakDecision = evidence.ResponseJailbreakDecision
	record.HallucinationScoreAvailable = evidence.HallucinationScoreAvailable
	record.HallucinationScoreKind = evidence.HallucinationScoreKind
	return nil
}
