package extproc

import (
	"encoding/json"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

func (r *OpenAIRouter) emitMetaRoutingFeedback(ctx *RequestContext, statusCode int) {
	if ctx == nil || ctx.MetaRoutingTrace == nil || ctx.MetaRoutingFeedbackWritten {
		return
	}

	record := buildMetaRoutingFeedbackRecord(ctx, statusCode)
	logging.LogEvent("meta_routing_feedback", map[string]interface{}{
		"request_id":      record.Observation.RequestID,
		"mode":            record.Mode,
		"final_decision":  record.Outcome.FinalDecisionName,
		"final_model":     record.Outcome.FinalModel,
		"response_status": record.Outcome.ResponseStatus,
		"pass_count":      ctx.MetaRoutingTrace.PassCount,
		"triggers":        ctx.MetaRoutingTrace.TriggerNames,
		"overturned":      ctx.MetaRoutingTrace.OverturnedDecision,
	})

	if r != nil && r.FeedbackRecorder != nil {
		if feedbackID, err := persistMetaRoutingFeedbackRecord(r.FeedbackRecorder, ctx, record); err != nil {
			logging.Errorf("Failed to persist meta-routing feedback: %v", err)
		} else {
			ctx.MetaRoutingFeedbackID = feedbackID
		}
	}

	ctx.MetaRoutingFeedbackWritten = true
}

func buildMetaRoutingFeedbackRecord(ctx *RequestContext, statusCode int) FeedbackRecord {
	action := FeedbackAction{
		Planned:  ctx.MetaRoutingTrace != nil && ctx.MetaRoutingTrace.FinalPlan != nil,
		Executed: ctx.MetaRoutingTrace != nil && ctx.MetaRoutingTrace.PassCount > 1,
		Plan:     nil,
	}
	if ctx.MetaRoutingTrace != nil {
		action.Plan = ctx.MetaRoutingTrace.FinalPlan
		action.ExecutedPassCount = ctx.MetaRoutingTrace.PassCount
		action.ExecutedActionTypes = metaRoutingTraceActionTypes(ctx.MetaRoutingTrace.FinalPlan)
		action.ExecutedSignalFamilies = append([]string(nil), ctx.MetaRoutingTrace.RefinedSignalFamilies...)
	}

	return FeedbackRecord{
		Mode: ctx.MetaRoutingTrace.Mode,
		Observation: FeedbackObservation{
			RequestID:      ctx.RequestID,
			RequestModel:   ctx.RequestModel,
			RequestQuery:   ctx.RequestQuery,
			PolicyProvider: cloneMetaRoutingPolicyDescriptor(ctx.MetaRoutingTrace.PolicyProvider),
			Trace:          ctx.MetaRoutingTrace,
		},
		Action: action,
		Outcome: FeedbackOutcome{
			FinalDecisionName:         ctx.MetaRoutingTrace.FinalDecisionName,
			FinalDecisionConfidence:   ctx.MetaRoutingTrace.FinalDecisionConfidence,
			FinalModel:                metaRoutingFinalModel(ctx),
			ResponseStatus:            statusCode,
			Streaming:                 ctx.ExpectStreamingResponse || ctx.IsStreamingResponse,
			CacheHit:                  ctx.VSRCacheHit,
			PIIBlocked:                ctx.PIIBlocked,
			HallucinationDetected:     ctx.HallucinationDetected,
			UnverifiedFactualResponse: ctx.UnverifiedFactualResponse,
			ResponseJailbreakDetected: ctx.ResponseJailbreakDetected,
			RAGBackend:                ctx.RAGBackend,
			RouterReplayID:            ctx.RouterReplayID,
			UserFeedbackSignals:       append([]string(nil), ctx.VSRMatchedUserFeedback...),
		},
	}
}

func metaRoutingTraceActionTypes(plan *RefinementPlan) []string {
	if plan == nil || len(plan.Actions) == 0 {
		return nil
	}
	types := make([]string, 0, len(plan.Actions))
	for _, action := range plan.Actions {
		if action.Type != "" {
			types = append(types, action.Type)
		}
	}
	return uniqueSortedStrings(types)
}

func metaRoutingFinalModel(ctx *RequestContext) string {
	if ctx == nil || ctx.MetaRoutingTrace == nil {
		return ""
	}
	if ctx.MetaRoutingTrace.FinalModel != "" {
		return ctx.MetaRoutingTrace.FinalModel
	}
	if ctx.RequestModel != "" {
		return ctx.RequestModel
	}
	return ""
}

func persistMetaRoutingFeedbackRecord(
	recorder *routerreplay.Recorder,
	ctx *RequestContext,
	record FeedbackRecord,
) (string, error) {
	encoded, err := json.Marshal(record)
	if err != nil {
		return "", err
	}

	rec := routerreplay.RoutingRecord{
		Timestamp:       time.Now().UTC(),
		RequestID:       ctx.RequestID,
		Decision:        record.Outcome.FinalDecisionName,
		Category:        ctx.VSRSelectedCategory,
		OriginalModel:   record.Observation.RequestModel,
		SelectedModel:   record.Outcome.FinalModel,
		ConfidenceScore: record.Outcome.FinalDecisionConfidence,
		SelectionMethod: ctx.VSRSelectionMethod,
		Signals:         replaySignalState(ctx),
		RequestBody:     string(encoded),
		ResponseStatus:  record.Outcome.ResponseStatus,
		FromCache:       record.Outcome.CacheHit,
		Streaming:       record.Outcome.Streaming,
	}
	return recorder.AddRecord(rec)
}

func (r *OpenAIRouter) collectMetaRoutingFeedbackRecords() []FeedbackRecord {
	stored := r.collectMetaRoutingFeedbackStoredRecords()
	feedback := make([]FeedbackRecord, 0, len(stored))
	for _, rec := range stored {
		feedback = append(feedback, rec.Record)
	}
	return feedback
}
