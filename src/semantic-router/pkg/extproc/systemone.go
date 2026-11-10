package extproc

import (
	"context"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// SystemOne uses this router generation's retained model lease. It bypasses
// recipe routing while preserving the native question and response contract.
func (r *OpenAIRouter) SystemOne(ctx context.Context, deployment string, body json.RawMessage) (modelservice.SystemOneResult, error) {
	if r == nil || r.signals == nil || r.signals.modelLease == nil {
		return modelservice.SystemOneResult{}, modelservice.ErrUnavailable
	}
	return r.signals.modelLease.SystemOne(ctx, deployment, body)
}
