package historyreset

import (
	"context"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

// TriggerRequest asks a topic-continuity producer to classify one request. The
// history is the resolved original conversation the action will transform, not
// the request as it arrived: a Responses request can carry earlier turns behind
// previous_response_id, and classifying a partial view would decide a topic
// change on a different conversation than the one being reset.
type TriggerRequest struct {
	Signal  string
	Binding string
	History contextcompression.ConversationHistory
}

// TriggerSource is the seam a topic-continuity signal implements to drive this
// action. The router resolves the configured signal through it after the
// permitted history is resolved and before the context stage runs. The action
// never classifies anything itself; it only consumes the typed result.
//
// The second result reports whether the source produced a usable result at
// all. A false value is missing evidence, which the configured failure mode
// then governs.
type TriggerSource interface {
	TopicContinuity(ctx context.Context, request TriggerRequest) (TriggerResult, bool)
}

// usableConfidence rejects values that cannot be compared against a threshold.
// NaN in particular fails every ordered comparison, so without this check an
// unusable confidence would slip past the minimum and authorize a removal.
func usableConfidence(confidence float64) bool {
	return !math.IsNaN(confidence) && !math.IsInf(confidence, 0) &&
		confidence >= 0 && confidence <= 1
}
