package store

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/topiccontinuity"

// TopicContinuityRecord is one topic-continuity rule's content-free replay
// receipt. It carries enums, counts, versions, scores, and cost only; never
// conversation text and never a content digest.
type TopicContinuityRecord struct {
	Signal                 string  `json:"signal"`
	SchemaVersion          string  `json:"schema_version"`
	EvaluatorVersion       string  `json:"evaluator_version"`
	Class                  string  `json:"class"`
	Confidence             float64 `json:"confidence"`
	Reason                 string  `json:"reason"`
	Fallback               bool    `json:"fallback"`
	Coverage               string  `json:"coverage"`
	HistorySource          string  `json:"history_source"`
	AssistantIncluded      bool    `json:"assistant_included"`
	ExcludedContentPresent bool    `json:"excluded_content_present"`
	FeatureCapReached      bool    `json:"feature_cap_reached"`
	MarkerAmbiguous        bool    `json:"marker_ambiguous"`
	PriorTurnsExamined     int     `json:"prior_turns_examined"`
	InputBytes             int     `json:"input_bytes"`
	LiveTerms              int     `json:"live_terms"`
	LiveProseTerms         int     `json:"live_prose_terms"`
	LexicalScore           float64 `json:"lexical_score"`
	EntityScore            float64 `json:"entity_score"`
	CombinedScore          float64 `json:"combined_score"`
	MaxRawScore            float64 `json:"max_raw_score"`
	MaxEntityScore         float64 `json:"max_entity_score"`
	// ClassifyMicros is this rule's Classify time; PrepareExtractMicros is its
	// policy group's shared Prepare plus Extract time.
	ClassifyMicros       int64 `json:"classify_us"`
	PrepareExtractMicros int64 `json:"prepare_extract_us"`
}

// NewTopicContinuityRecords maps evaluations, already in rule declaration
// order, to replay records in the same order.
func NewTopicContinuityRecords(evaluations []topiccontinuity.RuleEvaluation) []TopicContinuityRecord {
	if len(evaluations) == 0 {
		return nil
	}
	records := make([]TopicContinuityRecord, 0, len(evaluations))
	for _, evaluation := range evaluations {
		result := evaluation.Result
		features := result.Features
		records = append(records, TopicContinuityRecord{
			Signal:                 result.Signal,
			SchemaVersion:          result.SchemaVersion,
			EvaluatorVersion:       result.EvaluatorVersion,
			Class:                  string(result.Class),
			Confidence:             result.Confidence,
			Reason:                 string(result.Reason),
			Fallback:               result.Fallback,
			Coverage:               string(result.Coverage),
			HistorySource:          string(result.HistorySource),
			AssistantIncluded:      result.Scope.AssistantIncluded,
			ExcludedContentPresent: result.Scope.ExcludedContentPresent,
			FeatureCapReached:      result.Scope.FeatureCapReached,
			MarkerAmbiguous:        features.MarkerAmbiguous,
			PriorTurnsExamined:     features.PriorTurnsExamined,
			InputBytes:             features.InputBytes,
			LiveTerms:              features.LiveTerms,
			LiveProseTerms:         features.LiveProseTerms,
			LexicalScore:           features.LexicalScore,
			EntityScore:            features.EntityScore,
			CombinedScore:          features.CombinedScore,
			MaxRawScore:            features.MaxRawScore,
			MaxEntityScore:         features.MaxEntityScore,
			ClassifyMicros:         evaluation.ClassifyMicros,
			PrepareExtractMicros:   evaluation.PrepareExtractMicros,
		})
	}
	return records
}
