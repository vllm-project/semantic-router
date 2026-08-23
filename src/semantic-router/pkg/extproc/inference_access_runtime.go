package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const inferenceAdmissionLease = 15 * time.Minute

const inferenceSettlementTimeout = 10 * time.Second

// InferenceAccessRuntime is the process-owned Router-native access seam. A
// managed composition injects one instance into every router generation.
type InferenceAccessRuntime interface {
	Authenticate(context.Context, accessruntime.AuthenticationRequest) (accessruntime.Authentication, error)
	Authorize(context.Context, accessruntime.AuthorizationRequest) (accessruntime.Authorization, error)
	Discover(context.Context, accessruntime.DiscoveryRequest) (accessruntime.Discovery, error)
	DiscoverCatalog(context.Context, accessruntime.CatalogDiscoveryRequest) (accessruntime.CatalogDiscovery, error)
	Admit(context.Context, accessruntime.AdmissionRequest) (accessruntime.Admission, error)
	JournalDispatch(context.Context, accessruntime.DispatchJournalRequest) (quotaruntime.MutationResult, error)
	ReadAttemptEvidence(context.Context, accessruntime.AttemptEvidenceRequest) (accessruntime.AttemptEvidenceSnapshot, error)
	Settle(context.Context, accessruntime.SettlementRequest) (quotaruntime.FinalizationResult, error)
}

type inferenceRequestAccess struct {
	mu              sync.Mutex
	session         accessruntime.Session
	target          accessruntime.Target
	tenant          accessruntime.TenantContext
	source          accessruntime.AuthenticationSource
	admission       *accessruntime.Admission
	entrypoint      *config.EntrypointMapping
	rule            *config.EntrypointRule
	settlementModel string
	finalized       bool
	settlement      *inferenceSettlementPlan
	settlementRun   *inferenceSettlementRun
}

type inferenceSettlementRun struct {
	done    chan struct{}
	err     error
	waiters int
}

func (r *OpenAIRouter) managedInferenceAccessEnabled() bool {
	return r != nil && r.Config != nil && r.Config.Access.Enabled
}

// bindInferenceAuthentication loads the process-local opaque access session
// selected before a managed Router generation was acquired. The raw bearer is
// consumed by RouterService and never enters RequestContext.
func (r *OpenAIRouter) bindInferenceAuthentication(ctx *RequestContext) *ext_proc.ProcessingResponse {
	if ctx == nil {
		return r.createErrorResponse(http.StatusInternalServerError, "request context unavailable")
	}
	if ctx.ManagedDispatch == nil {
		ctx.ManagedDispatch = &managedRequestDispatch{requestID: ctx.RequestID}
	}
	if ctx.LooperRequest || !r.managedInferenceAccessEnabled() {
		return nil
	}
	removeHeaderValueCI(ctx, "authorization")
	authentication, ok := inferenceAuthenticationFromContext(ctx.TraceContext)
	if !ok {
		return r.createInferenceAccessError(quotaruntime.AdmissionUnauthenticated, nil)
	}
	ctx.InferenceAccess = &inferenceRequestAccess{
		session: authentication.session,
		tenant:  authentication.tenant,
		source:  authentication.source,
	}
	return nil
}

func (r *OpenAIRouter) authorizeInferenceTarget(
	ctx context.Context,
	request *RequestContext,
	model string,
) *ext_proc.ProcessingResponse {
	if !r.managedInferenceAccessEnabled() || request.LooperRequest {
		return nil
	}
	if r.InferenceAccess == nil {
		return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil)
	}
	state := request.InferenceAccess
	if state == nil || state.tenant.NamespaceID == "" {
		return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil)
	}
	model = strings.TrimSpace(model)
	if entrypoint := r.entrypointForAlias(model); entrypoint != nil {
		target := accessruntime.Target{
			ResourceType: accesscontrol.GrantResourceEntrypoint,
			ResourceID:   accesscontrol.ResourceID(entrypoint.ID),
			Permission:   accesscontrol.GrantPermissionInvoke,
		}
		checked, err := r.InferenceAccess.Authorize(ctx, accessruntime.AuthorizationRequest{
			Session: state.session, Target: target,
		})
		if err != nil || !checked.Result.Allowed() {
			return r.createInferenceAccessError(checked.Result.Disposition, nil)
		}
		if checked.Tenant.NamespaceID == "" {
			return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil)
		}
		claims := entrypointClaims(checked.Tenant.RoutingClaims)
		resolution, err := r.Config.ResolveEntrypoint(model, normalizedInferencePath(request), claims)
		if err != nil {
			return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil)
		}
		if resolution.Outcome != config.EntrypointResolveMatched || resolution.Recipe == nil {
			return r.createInferenceAccessError(quotaruntime.AdmissionForbidden, nil)
		}
		request.Routing.SelectRecipe(resolution.Recipe)
		state.target = target
		state.tenant = checked.Tenant
		state.entrypoint = resolution.Entrypoint
		state.rule = resolution.Rule
		return nil
	}

	params, exists := r.Config.ModelConfig[model]
	if !exists || strings.TrimSpace(params.ResourceID) == "" || params.ResourceRevision <= 0 {
		return r.createInferenceAccessError(quotaruntime.AdmissionForbidden, nil)
	}
	target := accessruntime.Target{
		ResourceType: accesscontrol.GrantResourceModel,
		ResourceID:   accesscontrol.ResourceID(params.ResourceID),
		Permission:   accesscontrol.GrantPermissionInvoke,
	}
	checked, err := r.InferenceAccess.Authorize(ctx, accessruntime.AuthorizationRequest{
		Session: state.session, Target: target,
	})
	if err != nil || !checked.Result.Allowed() {
		return r.createInferenceAccessError(checked.Result.Disposition, nil)
	}
	if checked.Tenant.NamespaceID == "" {
		return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil)
	}
	request.Routing.SelectPassthrough()
	state.target = target
	state.tenant = checked.Tenant
	return nil
}

func (r *OpenAIRouter) admitInferenceRequest(
	ctx context.Context,
	request *RequestContext,
	selectedModel string,
) *ext_proc.ProcessingResponse {
	if !r.managedInferenceAccessEnabled() || request.LooperRequest {
		return nil
	}
	state := request.InferenceAccess
	if state == nil || state.tenant.NamespaceID == "" || r.InferenceAccess == nil {
		return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil)
	}
	// An Entrypoint grant authorizes its complete immutable published action.
	// Internal Model assignments are implementation details of that virtual
	// Model and must not require callers to also hold direct-Model grants.
	// We still fail closed if the selected action references an unpinned model.
	candidates := r.inferenceCandidateModels(request, selectedModel)
	for _, model := range candidates {
		params, exists := r.Config.ModelConfig[model]
		if !exists || params.ResourceID == "" || params.ResourceRevision <= 0 {
			return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil)
		}
	}

	digest := inferenceRequestDigest(request, state.target)
	admission, err := r.InferenceAccess.Admit(ctx, accessruntime.AdmissionRequest{
		Session:       state.session,
		Target:        state.target,
		AdmissionID:   uuid.NewString(),
		RequestDigest: digest,
		LeaseDuration: inferenceAdmissionLease,
	})
	if err != nil || !admission.Result.Allowed() {
		return r.createInferenceAccessError(admission.Result.Disposition, admission.Result.RetryAt)
	}
	if admission.Tenant.NamespaceID == "" {
		return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil)
	}
	settlementModel := strings.TrimSpace(selectedModel)
	if settlementModel == "" && len(candidates) > 0 {
		settlementModel = candidates[0]
	}
	state.mu.Lock()
	state.admission = &admission
	state.tenant = admission.Tenant
	state.settlementModel = settlementModel
	state.mu.Unlock()
	return nil
}

func (r *OpenAIRouter) inferenceCandidateModels(request *RequestContext, selected string) []string {
	seen := make(map[string]struct{})
	appendModel := func(value string) {
		value = strings.TrimSpace(value)
		if value != "" {
			seen[value] = struct{}{}
		}
	}
	appendModel(selected)
	if strings.TrimSpace(selected) == "" && r != nil && r.Config != nil {
		appendModel(r.Config.DefaultModel)
	}
	if request != nil {
		if _, exists := r.Config.ModelConfig[request.RequestModel]; exists {
			appendModel(request.RequestModel)
		}
	}
	if request != nil && request.VSRSelectedDecision != nil {
		decision := request.VSRSelectedDecision
		for _, ref := range decision.ModelRefs {
			appendModel(ref.Model)
		}
		for _, iteration := range decision.CandidateIterations {
			for _, ref := range iteration.Models {
				appendModel(ref.Model)
			}
		}
		if decision.Algorithm != nil {
			if decision.Algorithm.Prompt != nil {
				appendModel(decision.Algorithm.Prompt.Model)
			}
			if decision.Algorithm.ReMoM != nil {
				appendModel(decision.Algorithm.ReMoM.SynthesisModel)
			}
			if decision.Algorithm.Fusion != nil {
				appendModel(decision.Algorithm.Fusion.Model)
				for _, model := range decision.Algorithm.Fusion.AnalysisModels {
					appendModel(model)
				}
			}
		}
	}
	models := make([]string, 0, len(seen))
	for model := range seen {
		models = append(models, model)
	}
	sort.Strings(models)
	return models
}

func (r *OpenAIRouter) entrypointForAlias(alias string) *config.EntrypointMapping {
	if r == nil || r.Config == nil {
		return nil
	}
	for index := range r.Config.Entrypoints {
		for _, candidate := range r.Config.Entrypoints[index].ModelNames {
			if candidate == alias {
				return &r.Config.Entrypoints[index]
			}
		}
	}
	return nil
}

func entrypointClaims(values map[string]routingsnapshot.ClaimValue) map[string]config.EntrypointClaimValue {
	result := make(map[string]config.EntrypointClaimValue, len(values))
	for name, value := range values {
		result[name] = config.EntrypointClaimValue{
			Kind: value.Kind, String: value.String, Boolean: value.Boolean, Integer: value.Integer,
		}
	}
	return result
}

func normalizedInferencePath(ctx *RequestContext) string {
	if ctx == nil {
		return "/v1/chat/completions"
	}
	return normalizeRequestPath(headerValueCI(ctx, ":path"))
}

func inferenceRequestDigest(ctx *RequestContext, target accessruntime.Target) string {
	hash := sha256.New()
	_, _ = hash.Write([]byte("vllm-sr/inference-admission/v1\x00"))
	_, _ = hash.Write([]byte(target.ResourceType))
	_, _ = hash.Write([]byte{0})
	_, _ = hash.Write([]byte(target.ResourceID))
	_, _ = hash.Write([]byte{0})
	if ctx != nil && ctx.SemanticRequest != nil {
		if request, err := cache.MarshalSemanticRequest(*ctx.SemanticRequest); err == nil {
			_, _ = hash.Write(request)
		}
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func (r *OpenAIRouter) createInferenceAccessError(
	disposition quotaruntime.AdmissionDisposition,
	retryAt *time.Time,
) *ext_proc.ProcessingResponse {
	switch disposition {
	case quotaruntime.AdmissionUnauthenticated:
		return r.createErrorResponse(http.StatusUnauthorized, "invalid or missing API key")
	case quotaruntime.AdmissionForbidden:
		return r.createErrorResponse(http.StatusNotFound, "model not found")
	case quotaruntime.AdmissionRateLimited:
		response := r.createErrorResponse(http.StatusTooManyRequests, "quota exceeded")
		if retryAt != nil {
			seconds := int64(time.Until(*retryAt).Seconds())
			if seconds < 1 {
				seconds = 1
			}
			if immediate := response.GetImmediateResponse(); immediate != nil {
				immediate.Headers.SetHeaders = append(immediate.Headers.SetHeaders, retryAfterHeader(seconds))
			}
		}
		return response
	default:
		return r.createErrorResponse(http.StatusServiceUnavailable, "inference access is temporarily unavailable")
	}
}

func inferenceAccessDisposition(response *ext_proc.ProcessingResponse) int {
	if response == nil || response.GetImmediateResponse() == nil {
		return 0
	}
	return int(response.GetImmediateResponse().Status.Code)
}

func retryAfterHeader(seconds int64) *core.HeaderValueOption {
	return &core.HeaderValueOption{Header: &core.HeaderValue{Key: "retry-after", Value: fmt.Sprintf("%d", seconds)}}
}
