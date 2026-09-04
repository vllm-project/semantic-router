package testcases

import (
	"context"
	"fmt"
	"net/http"
)

const (
	pendingCancelConflict = "evaluation resource conflict: run cannot be cancelled from pending"
	pendingReportConflict = "evaluation resource conflict: evaluation report is available only for completed runs"
)

func verifyPendingEvaluationCancellationGuard(
	ctx context.Context,
	client *http.Client,
	baseURL, token string,
) error {
	pending, err := createEvaluationRun(
		ctx, client, baseURL, token, newEvaluationClientRequestID(), "cancel-contract", 43, "",
	)
	if err != nil {
		return err
	}
	if pending.Status != "pending" {
		return fmt.Errorf("cancellation guard run status = %q, want pending", pending.Status)
	}
	url := fmt.Sprintf("%s/api/evaluation/v1/runs/%s/cancel", baseURL, pending.ID)
	var response dashboardEvaluationErrorResponse
	if err := evaluationJSON(
		ctx, client, http.MethodPost, url, token, map[string]interface{}{},
		&response, http.StatusConflict,
	); err != nil {
		return err
	}
	if response.Error.Message != pendingCancelConflict {
		return fmt.Errorf("pending cancellation was rejected for the wrong reason: %s", response.Error.Message)
	}
	if err := verifyEvaluationRunRemainsPending(ctx, client, baseURL, token, pending); err != nil {
		return err
	}
	return verifyPendingEvaluationReportUnavailable(ctx, client, baseURL, token, pending.ID)
}

func verifyEvaluationRunRemainsPending(
	ctx context.Context,
	client *http.Client,
	baseURL, token string,
	pending dashboardEvaluationRun,
) error {
	var unchanged dashboardEvaluationRun
	if err := evaluationJSON(
		ctx, client, http.MethodGet, baseURL+"/api/evaluation/v1/runs/"+pending.ID,
		token, nil, &unchanged, http.StatusOK,
	); err != nil {
		return err
	}
	if unchanged.ID != pending.ID || unchanged.ClientRequestID != pending.ClientRequestID || unchanged.Status != "pending" {
		return fmt.Errorf(
			"rejected cancellation changed pending run identity or status: id=%q request=%q status=%q",
			unchanged.ID, unchanged.ClientRequestID, unchanged.Status,
		)
	}
	if unchanged.StartedAt != nil || unchanged.CompletedAt != nil || unchanged.Progress.Percent != 0 {
		return fmt.Errorf("rejected cancellation added execution state to the pending run")
	}
	return nil
}

func verifyPendingEvaluationReportUnavailable(
	ctx context.Context,
	client *http.Client,
	baseURL, token, runID string,
) error {
	var response dashboardEvaluationErrorResponse
	if err := evaluationJSON(
		ctx, client, http.MethodGet, baseURL+"/api/evaluation/v1/runs/"+runID+"/report",
		token, nil, &response, http.StatusConflict,
	); err != nil {
		return err
	}
	if response.Error.Message != pendingReportConflict {
		return fmt.Errorf("pending run exposed the wrong report boundary: %s", response.Error.Message)
	}
	return nil
}
