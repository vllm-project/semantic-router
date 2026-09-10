package contextcompression

import (
	"context"
	"errors"
)

var errInvalidCompressionEdit = errors.New("compression violated the transformation contract")

// Apply retains the existing compressor and records all outcomes in the
// request-owned transformation plan, after any enabled history steps.
func (s *Service) Apply(ctx context.Context, request Request) ServiceResult {
	if request.Request == nil {
		return s.apply(ctx, request)
	}
	plan := &request.Request.Transformations
	if plan.terminal != nil {
		return ServiceResult{Request: request.Request, Failure: plan.terminal}
	}
	for _, receipt := range plan.receipts {
		if receipt.Kind == TransformCompress {
			return ServiceResult{Request: request.Request, Plan: Plan{SkipReason: SkipNoTargets}}
		}
	}
	result := s.apply(ctx, request)
	if plan.last != TransformCompress {
		receipt := stepReceipt(TransformCompress)
		receipt.Status, receipt.Reason = TransformationSkipped, string(result.Plan.SkipReason)
		if result.Failure != nil {
			receipt.Status, receipt.Reason = TransformationFailed, "compression_failed"
			if request.Policy.FailureMode == FailureClosed {
				plan.terminal = result.Failure
			}
		}
		plan.last = TransformCompress
		plan.receipts = append(plan.receipts, receipt)
	}
	return result
}

func (s *Service) commitCompression(ctx context.Context, request Request, counter TokenCounter, result ServiceResult, candidates []plannedCandidate) ServiceResult {
	edits := TransformationEdits{}
	for _, candidate := range candidates {
		edits.ReplaceText = append(edits.ReplaceText, TextReplacement{MessageID: candidate.plan.MessageIndex, BlockID: candidate.plan.BlockIndex, Text: candidate.replacementText})
	}
	receipt, err := request.Request.applyStep(ctx, TransformationStep{
		Kind: TransformCompress, Enabled: true, FailureMode: request.Policy.FailureMode,
		Propose: func(context.Context, TransformationView) (TransformationEdits, error) { return edits, nil },
	}, func() error {
		after, _ := counter.CountRequest(request.Model, request.Request)
		result.TokensAfter = after
		if after <= 0 || after >= result.TokensBefore {
			return errNonReducing
		}
		return nil
	})
	if err != nil || receipt.Status == TransformationFailed {
		if err == nil {
			err = errInvalidCompressionEdit
		}
		s.failures.Add(1)
		return s.failureResult(request, result.Plan, err)
	}
	if receipt.Status != TransformationApplied {
		result.Applied = false
		result.TokensAfter = result.TokensBefore
		result.BlocksCompressed, result.MessagesCompressed = 0, 0
		result.OmittedChunks, result.JSONBlocks = 0, 0
		result.RecoveryKeys = nil
		if receipt.Reason == "budget_not_reduced" {
			result.Plan.Quality = "rejected_non_reducing"
		}
	}
	return s.finalizeResult(request, counter, result)
}
