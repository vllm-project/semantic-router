package modelservice

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

type (
	meterKey           struct{}
	deploymentMeterKey struct{}
)

// ModelCall is one dispatched task, before a fused response is split among
// its callers. Missing usage remains unknown, including on failed calls.
type ModelCall struct {
	Deployment   string
	Kind         string
	Model        string
	InputTokens  int
	OutputTokens int
	HasUsage     bool
}

// RequestMeter belongs to one request. Its lock is never shared by requests.
// Snapshots also work after cancellation, so failures retain their spend.
type RequestMeter struct {
	mu    sync.Mutex
	calls []ModelCall
}

// WithRequestMeter starts accounting only for descendants of this context.
// Startup, probes and background work do not install a meter.
func WithRequestMeter(ctx context.Context) (context.Context, *RequestMeter) {
	meter := &RequestMeter{}
	return context.WithValue(ctx, meterKey{}, meter), meter
}

// WithMeterDeployment identifies the deployment used by a metered call.
func WithMeterDeployment(ctx context.Context, name string) context.Context {
	return context.WithValue(ctx, deploymentMeterKey{}, name)
}

func (m *RequestMeter) Calls() []ModelCall {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]ModelCall(nil), m.calls...)
}

// RecordModelCall reports usage from a dispatched model call to its request.
func RecordModelCall(ctx context.Context, call ModelCall) {
	meter, _ := ctx.Value(meterKey{}).(*RequestMeter)
	if meter == nil {
		return
	}
	if call.Deployment == "" {
		call.Deployment, _ = ctx.Value(deploymentMeterKey{}).(string)
	}
	meter.mu.Lock()
	meter.calls = append(meter.calls, call)
	meter.mu.Unlock()
}

func reportRuntimeCall(ctx context.Context, task api.BundleTask, result api.BundleResult) {
	call := ModelCall{}
	switch {
	case task.Decisions != nil:
		call.Kind, call.Model = "decision_model", deref(task.Decisions.Model)
		if body := result.Decisions; body != nil {
			call.Model, call.InputTokens, call.OutputTokens = body.Model, body.Usage.InputTokens, body.Usage.OutputTokens
		}
	case task.Embeddings != nil:
		call.Kind, call.Model = "embeddings", deref(task.Embeddings.Model)
		if body := result.Embeddings; body != nil {
			call.Model, call.InputTokens = body.Model, body.Usage.PromptTokens
		}
	case task.Classify != nil:
		call.Kind, call.Model = "classify", deref(task.Classify.Model)
		if body := result.Classify; body != nil {
			call.Model, call.InputTokens, call.OutputTokens = body.Model, body.Usage.InputTokens, body.Usage.OutputTokens
		}
	case task.Rerank != nil:
		call.Kind, call.Model = "rerank", deref(task.Rerank.Model)
		if body := result.Rerank; body != nil {
			call.Model, call.InputTokens, call.OutputTokens = body.Model, body.Usage.InputTokens, body.Usage.OutputTokens
		}
	default:
		return
	}
	// The generated runtime contract represents absent usage as a zero value.
	// A zero-token answer cannot establish that the dispatched call was free.
	call.HasUsage = call.InputTokens > 0 || call.OutputTokens > 0
	RecordModelCall(ctx, call)
}

func reportRemoteCall(ctx context.Context, deployment, model string, data []byte) {
	var body struct {
		Model string `json:"model"`
		Usage *struct {
			Input  int `json:"input_tokens"`
			Output int `json:"output_tokens"`
		} `json:"usage"`
	}
	_ = json.Unmarshal(data, &body)
	call := ModelCall{Deployment: deployment, Kind: "systemone", Model: model}
	if body.Model != "" {
		call.Model = body.Model
	}
	if body.Usage != nil {
		call.InputTokens, call.OutputTokens, call.HasUsage = body.Usage.Input, body.Usage.Output, true
	}
	RecordModelCall(ctx, call)
}
