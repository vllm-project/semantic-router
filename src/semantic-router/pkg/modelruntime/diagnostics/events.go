// Package diagnostics reports prepared model execution without model inputs.
package diagnostics

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func ErrorClass(err error) string {
	if err == nil {
		return ""
	}
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return "deadline"
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.Is(err, admission.ErrQueueFull):
		return "queue_full"
	case errors.Is(err, binding.ErrClosed):
		return "closed"
	case errors.Is(err, binding.ErrCapability):
		return "capability"
	case errors.Is(err, binding.ErrInputLimit), errors.Is(err, tasks.ErrTokenSpansTruncated):
		return "input_limit"
	case errors.Is(err, binding.ErrInvalidInput):
		return "invalid_input"
	case errors.Is(err, binding.ErrInvalidResult):
		return "invalid_result"
	}
	var remote *connector.Error
	if errors.As(err, &remote) {
		return "remote_" + string(remote.Kind)
	}
	return "model_execution"
}

func Observe(event binding.Event) {
	limits := event.Capability.Limits
	payload := map[string]interface{}{
		"recipe": event.Identity.Recipe, "binding": event.Identity.Name, "deployment": event.Identity.Deployment,
		"task": event.Identity.Contract, "adapter": event.Identity.Adapter, "provider": event.Capability.Provider,
		"device": event.Capability.Device, "precision": event.Capability.Precision, "state": event.State,
		"model_tokens": limits.ModelTokens, "task_tokens": limits.TaskTokens, "deployment_tokens": limits.DeploymentTokens,
		"effective_tokens": limits.EffectiveTokens(), "overflow": limits.Overflow, "error_class": ErrorClass(event.Error),
		"duration_ms": float64(event.Duration.Microseconds()) / 1000,
	}
	if event.Input != nil {
		payload["input_tokens"] = event.Input.OriginalTokens
		payload["processed_tokens"] = event.Input.ProcessedTokens
		payload["truncated"] = event.Input.Truncated
	}
	if event.State == "call" {
		wait := event.AdmissionWait
		outcome := "admitted"
		if !event.Executed {
			wait = event.Duration
			outcome = "canceled"
			if errors.Is(event.Error, admission.ErrQueueFull) {
				outcome = "shed"
			}
		}
		// Rejected recipe lookups and invalid inputs never entered admission.
		if event.Executed || errors.Is(event.Error, admission.ErrQueueFull) || errors.Is(event.Error, context.Canceled) || errors.Is(event.Error, context.DeadlineExceeded) {
			metrics.RecordModelAdmission(event.Identity.Deployment, outcome, wait.Seconds())
		}
		logging.ComponentDebugEvent("modelruntime", "model_binding_call", payload)
		return
	}
	logging.ComponentEvent("modelruntime", "model_binding_"+event.State, payload)
}
