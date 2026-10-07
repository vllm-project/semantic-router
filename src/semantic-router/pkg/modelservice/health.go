package modelservice

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// RuntimeAPIMajor is the major version of the runtime contract
// (src/model-runtime/vllm_srun/api/openapi.yaml info.version) this
// client was generated from. A runtime that serves another major is refused.
const RuntimeAPIMajor = "2"

// processHealth is a runtime process's /health: its own state and, when it
// serves several models, each model's state.
type processHealth struct {
	status     string
	reason     string
	apiVersion string
	models     map[string]modelHealth
}

type modelHealth struct {
	status string
	reason string
}

// incompatible reports why the process's contract version is not the one
// this client speaks, or "" when it is.
func (h processHealth) incompatible() string {
	if major, _, _ := strings.Cut(h.apiVersion, "."); major == RuntimeAPIMajor {
		return ""
	}
	served := h.apiVersion
	if served == "" {
		served = "no api_version"
	}
	return fmt.Sprintf("the runtime serves contract %s; this router speaks %s.x", served, RuntimeAPIMajor)
}

// stateOf returns a model's state. A process that serves one model may
// report only its own state, which then is the model's.
func (h processHealth) stateOf(model string, served int) (string, string) {
	if reason := h.incompatible(); reason != "" {
		return "incompatible", reason
	}
	if state, ok := h.models[model]; ok {
		return state.status, state.reason
	}
	if served == 1 && len(h.models) == 0 {
		return h.status, h.reason
	}
	return "missing", "the runtime does not serve this model"
}

// health reads /health; both 200 and 503 carry a state.
func (c *Client) health(ctx context.Context) (processHealth, error) {
	response, err := c.api.GetHealthWithResponse(ctx)
	if err != nil {
		return processHealth{}, fmt.Errorf("%w: %w", ErrFailed, err)
	}
	for _, body := range []*api.Health{response.JSON200, response.JSON503} {
		if body == nil {
			continue
		}
		health := processHealth{status: string(body.Status), reason: deref(body.Reason), apiVersion: body.ApiVersion}
		if body.Models != nil {
			health.models = make(map[string]modelHealth, len(*body.Models))
			for name, model := range *body.Models {
				health.models[name] = modelHealth{status: string(model.Status), reason: deref(model.Reason)}
			}
		}
		return health, nil
	}
	return processHealth{}, fmt.Errorf("%w: /health returned %s", ErrFailed, response.Status())
}
