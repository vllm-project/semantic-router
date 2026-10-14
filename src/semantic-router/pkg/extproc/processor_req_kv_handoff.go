package extproc

import (
	"context"
	"encoding/json"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/kvtransfer"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// KVDispatch names the route after model selection. Implementations must resolve
// exact identities and capabilities from trusted deployment state, never aliases
// or user-supplied request fields. B1 supplies the source lookup separately.
type KVDispatch struct {
	Principal         string
	SessionID         string
	SessionProvenance string
	PreviousModel     string
	TargetModel       string
	BackendName       string
	Turn              int
}

// KVHandoffPlanner bridges trusted deployment facts to the B3 coordinator.
// It must return a hint only after B2/B3 validation. It is installed before
// serving traffic; an absent adapter leaves every request on normal prefill.
type KVHandoffPlanner interface {
	PlanDispatch(context.Context, KVDispatch) (*kvtransfer.Hint, kvtransfer.Reason)
}

// KVTargetResolver supplies pinned target identity and explicit connector
// capabilities for a concrete backend. Missing facts disable reuse.
type KVTargetResolver interface {
	ResolveKVTarget(model, backend string) (kvtransfer.ModelIdentity, bool, bool, bool)
}

// CoordinatedKVHandoff is the dispatch adapter for the B3 coordinator.
// Registry/config initialization is supplied by the deployment integration.
type CoordinatedKVHandoff struct {
	Coordinator *kvtransfer.Coordinator
	Targets     KVTargetResolver
}

func (a *CoordinatedKVHandoff) PlanDispatch(ctx context.Context, d KVDispatch) (*kvtransfer.Hint, kvtransfer.Reason) {
	if a == nil || a.Targets == nil {
		return nil, kvtransfer.ReasonDisabled
	}
	target, _, load, ok := a.Targets.ResolveKVTarget(d.TargetModel, d.BackendName)
	if !ok {
		return nil, kvtransfer.ReasonUnavailableBackend
	}
	return a.Coordinator.Plan(ctx, kvtransfer.Request{
		AuthenticatedPrincipal: d.Principal, SessionID: d.SessionID,
		SessionProvenance: d.SessionProvenance, Target: target,
		// Only the source resolver can confirm export capability.
		TargetCanLoad: load, Now: time.Now(),
	}, d.Turn)
}

// encodeKVHandoff runs after provider encoding so vLLM transfer parameters do
// not become model input or leak into another provider's wire format.
func (r *OpenAIRouter) encodeKVHandoff(body []byte, format llmprotocol.WireFormat, ctx *RequestContext) ([]byte, error) {
	if r == nil || r.KVHandoff == nil || ctx == nil || format != llmprotocol.OpenAIChatV1 {
		return body, nil
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, err
	}
	// This extension is router-owned whenever the handoff adapter is enabled.
	// Caller hints must never authorize a cross-tenant cache read.
	_, hadHint := wire["kv_transfer_params"]
	delete(wire, "kv_transfer_params")
	var hint *kvtransfer.Hint
	if !ctx.LooperRequest && ctx.AuthenticatedPrincipal != "" && ctx.SessionID != "" &&
		(ctx.SessionProvenance == SessionProvenanceHeader || ctx.SessionProvenance == SessionProvenanceResponseAPI) &&
		ctx.PreviousModel != "" && ctx.PreviousModel != ctx.RequestModel {
		hint, _ = r.KVHandoff.PlanDispatch(selectionRequestContext(ctx), KVDispatch{
			Principal: ctx.AuthenticatedPrincipal, SessionID: ctx.SessionID,
			SessionProvenance: string(ctx.SessionProvenance), PreviousModel: ctx.PreviousModel,
			TargetModel: ctx.RequestModel, BackendName: ctx.primaryBackendName, Turn: ctx.TurnIndex,
		})
	}
	if hint != nil && (hint.Namespace != cache.UserScopeNamespace(ctx.AuthenticatedPrincipal) ||
		!cache.UserScopeSecretConfigured() || hint.CacheID == "" || hint.MapperID == "") {
		hint = nil
	}
	if hint == nil {
		if !hadHint {
			return body, nil
		}
		return json.Marshal(wire)
	}
	value, err := json.Marshal(map[string]string{"namespace": hint.Namespace, "cache_id": hint.CacheID, "mapper_id": hint.MapperID})
	if err != nil {
		return nil, err
	}
	wire["kv_transfer_params"] = value
	return json.Marshal(wire)
}
