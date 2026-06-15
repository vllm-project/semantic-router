package handlers

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

func reportBuilderNLProgress(
	reporter builderNLProgressReporter,
	phase string,
	level string,
	message string,
	attempt int,
) {
	emitBuilderNLProgress(reporter, phase, level, message, attempt, "stage", 0)
}

func reportBuilderNLHeartbeat(
	reporter builderNLProgressReporter,
	phase string,
	message string,
	attempt int,
	elapsedSeconds int,
) {
	emitBuilderNLProgress(reporter, phase, builderNLProgressInfo, message, attempt, "heartbeat", elapsedSeconds)
}

func emitBuilderNLProgress(
	reporter builderNLProgressReporter,
	phase string,
	level string,
	message string,
	attempt int,
	kind string,
	elapsedSeconds int,
) {
	if reporter == nil {
		return
	}

	reporter(BuilderNLProgressEvent{
		Phase:          phase,
		Level:          level,
		Message:        strings.TrimSpace(message),
		Attempt:        attempt,
		Kind:           kind,
		ElapsedSeconds: elapsedSeconds,
		Timestamp:      time.Now().UnixMilli(),
	})
}

func runBuilderNLModelCallWithProgress(
	ctx context.Context,
	reporter builderNLProgressReporter,
	phase string,
	attempt int,
	timeout time.Duration,
	call func() (string, error),
) (string, error) {
	if reporter == nil {
		return call()
	}

	start := time.Now()
	done := make(chan struct{})
	reportBuilderNLProgress(
		reporter,
		phase,
		builderNLProgressInfo,
		fmt.Sprintf("Waiting for model response (timeout %s).", timeout.Round(time.Second)),
		attempt,
	)

	go func() {
		ticker := time.NewTicker(builderNLHeartbeatInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-done:
				return
			case <-ticker.C:
				elapsed := int(time.Since(start).Seconds())
				reportBuilderNLHeartbeat(
					reporter,
					phase,
					fmt.Sprintf(
						"Still waiting for model response (%ds elapsed of %ds timeout).",
						elapsed,
						int(timeout.Seconds()),
					),
					attempt,
					elapsed,
				)
			}
		}
	}()

	content, err := call()
	close(done)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			reportBuilderNLProgress(
				reporter,
				phase,
				builderNLProgressError,
				fmt.Sprintf("Model call timed out after %s.", timeout.Round(time.Second)),
				attempt,
			)
		} else {
			reportBuilderNLProgress(reporter, phase, builderNLProgressError, fmt.Sprintf("Model call failed: %s", err), attempt)
		}
		return "", err
	}

	reportBuilderNLProgress(reporter, phase, builderNLProgressSuccess, fmt.Sprintf("Model response received after %s.", time.Since(start).Round(time.Second)), attempt)
	return content, nil
}
