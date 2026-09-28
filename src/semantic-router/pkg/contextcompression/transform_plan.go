package contextcompression

import (
	"context"
	"errors"
	"fmt"
)

var errNonReducing = errors.New("transformation did not reduce the context budget")

// ApplySteps validates the declared order before executing any step. Encoding
// belongs to the caller and must happen only after this plan has completed.
func (request *RequestIR) ApplySteps(ctx context.Context, steps []TransformationStep) error {
	last := TransformationKind(0)
	for _, step := range steps {
		if step.Kind <= last || step.Kind >= TransformCompress {
			return fmt.Errorf("invalid context transformation order")
		}
		if step.Kind <= request.Transformations.last && !request.Transformations.completed(step.Kind) {
			return fmt.Errorf("context transformation added after a later step")
		}
		last = step.Kind
	}
	for _, step := range steps {
		if step.Enabled && !request.Transformations.completed(step.Kind) {
			request.Transformations.historyEnabled = true
		}
		if _, err := request.applyStep(ctx, step, nil); err != nil {
			return err
		}
	}
	return nil
}

// applyStep is also the integration seam used by the existing compressor.
// accept runs after a validated tentative text edit, with rollback on rejection.
func (request *RequestIR) applyStep(
	ctx context.Context,
	step TransformationStep,
	accept func() error,
) (TransformationReceipt, error) {
	plan := &request.Transformations
	if plan.terminal != nil {
		return TransformationReceipt{}, plan.terminal
	}
	for _, receipt := range plan.receipts {
		if receipt.Kind == step.Kind {
			return receipt, nil
		}
	}
	if step.Kind <= plan.last || step.Kind > TransformCompress {
		return TransformationReceipt{}, fmt.Errorf("invalid context transformation order")
	}
	receipt := stepReceipt(step.Kind)
	plan.last = step.Kind
	if !step.Enabled {
		receipt.Status, receipt.Reason = TransformationSkipped, "disabled"
	} else {
		receipt = request.executeStep(ctx, step, receipt, accept)
	}
	plan.receipts = append(plan.receipts, receipt)
	if receipt.Status == TransformationFailed && step.FailureMode == FailureClosed {
		plan.terminal = fmt.Errorf("context transformation %d failed: %s", step.Kind, receipt.Reason)
		return receipt, plan.terminal
	}
	return receipt, nil
}

func stepReceipt(kind TransformationKind) TransformationReceipt {
	receipt := TransformationReceipt{Kind: kind, Input: InputEligibleHistory, Output: OutputRemoveMessages}
	if kind == TransformCompress {
		receipt.Input, receipt.Output = InputConfiguredBlocks, OutputReplaceText
	}
	return receipt
}

func (request *RequestIR) executeStep(
	ctx context.Context,
	step TransformationStep,
	receipt TransformationReceipt,
	accept func() error,
) TransformationReceipt {
	receipt.Status, receipt.Reason = TransformationFailed, "policy_failed"
	if step.Propose == nil || ctx.Err() != nil {
		return receipt
	}
	edits, err := step.Propose(ctx, request.TransformationView())
	if err != nil || ctx.Err() != nil {
		return receipt
	}
	if err := request.validateEdits(step.Kind, edits); err != nil {
		receipt.Reason = "invariant_violation"
		return receipt
	}
	if len(edits.RemoveMessages) == 0 && len(edits.ReplaceText) == 0 {
		receipt.Status, receipt.Reason = TransformationSkipped, "no_changes"
		return receipt
	}
	undo := request.commitEdits(edits)
	if accept != nil {
		if err := accept(); err != nil {
			undo()
			receipt.Status, receipt.Reason = TransformationSkipped, "budget_not_reduced"
			return receipt
		}
	}
	receipt.Status, receipt.Reason = TransformationApplied, ""
	receipt.MessagesRemoved = len(edits.RemoveMessages)
	receipt.BlocksReplaced = len(edits.ReplaceText)
	return receipt
}

func (plan *TransformationPlan) completed(kind TransformationKind) bool {
	for _, receipt := range plan.receipts {
		if receipt.Kind == kind {
			return true
		}
	}
	return false
}

// SkipCompression seals the final stage when the plugin is disabled or bypassed.
func (request *RequestIR) SkipCompression(ctx context.Context) error {
	_, err := request.applyStep(ctx, TransformationStep{Kind: TransformCompress}, nil)
	return err
}
