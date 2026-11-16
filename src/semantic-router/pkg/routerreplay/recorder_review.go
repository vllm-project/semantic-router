package routerreplay

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"

type (
	Review           = store.Review
	ReviewAssessment = store.ReviewAssessment
	ReviewAction     = store.ReviewAction
)

// SettleReview stores the review a record ended with. Its latency is the
// duration the record's lifecycle recorded, so call it after
// FinalizeLifecycle.
func (r *Recorder) SettleReview(id string, review Review) error {
	if record, found, err := r.getRecord(id); err != nil {
		return err
	} else if found {
		review.LatencyMS = record.DurationMS
	}
	ctx, cancel := r.replayOperationContext()
	defer cancel()
	return r.storage.UpdateReview(ctx, id, review)
}
