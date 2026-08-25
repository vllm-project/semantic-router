package extproc

import (
	"context"
	"errors"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const inferenceAdmissionHeartbeatInterval = inferenceAdmissionMinimumLease / 3

func (r *OpenAIRouter) startInferenceAdmissionHeartbeat(
	request *RequestContext,
	admission accessruntime.Admission,
) {
	r.startInferenceAdmissionHeartbeatEvery(request, admission, inferenceAdmissionHeartbeatInterval)
}

func (r *OpenAIRouter) startInferenceAdmissionHeartbeatEvery(
	request *RequestContext,
	admission accessruntime.Admission,
	interval time.Duration,
) {
	if r == nil || r.InferenceAccess == nil || request == nil || request.InferenceAccess == nil || interval <= 0 ||
		admission.Result.PlanDigest == "" {
		return
	}
	base := request.TraceContext
	if base == nil {
		base = context.Background()
	}
	heartbeatContext, cancel := context.WithCancel(base)
	done := make(chan struct{})
	state := request.InferenceAccess
	state.mu.Lock()
	if state.heartbeatCancel != nil || state.finalized {
		state.mu.Unlock()
		cancel()
		close(done)
		return
	}
	state.heartbeatCancel = cancel
	state.heartbeatDone = done
	state.mu.Unlock()

	go func() {
		defer close(done)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-heartbeatContext.Done():
				return
			case <-ticker.C:
				callContext, callCancel := context.WithTimeout(heartbeatContext, heartbeatCallTimeout(interval))
				result, err := r.InferenceAccess.Heartbeat(callContext, admission)
				callCancel()
				if err != nil {
					if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
						logging.ComponentWarnEvent("extproc", "inference_admission_heartbeat_failed", map[string]interface{}{
							"request_id": request.RequestID,
							"error":      err.Error(),
						})
					}
					continue
				}
				if result.Stopped {
					return
				}
			}
		}
	}()
}

func heartbeatCallTimeout(interval time.Duration) time.Duration {
	const maximum = 10 * time.Second
	if interval < maximum {
		return interval
	}
	return maximum
}

func stopInferenceAdmissionHeartbeat(state *inferenceRequestAccess) {
	if state == nil {
		return
	}
	state.mu.Lock()
	cancel := state.heartbeatCancel
	done := state.heartbeatDone
	state.heartbeatCancel = nil
	state.heartbeatDone = nil
	state.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	if done != nil {
		<-done
	}
}
