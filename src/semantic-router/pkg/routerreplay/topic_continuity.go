package routerreplay

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/topiccontinuity"
)

// NewTopicContinuityRecords maps topic-continuity evaluations, in rule
// declaration order, to content-free replay records.
func NewTopicContinuityRecords(evaluations []topiccontinuity.RuleEvaluation) []TopicContinuityRecord {
	return store.NewTopicContinuityRecords(evaluations)
}
