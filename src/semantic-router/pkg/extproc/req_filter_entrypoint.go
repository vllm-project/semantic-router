package extproc

import (
	"context"
	"strings"

	"github.com/tidwall/gjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// resolveEntrypointForRequest resolves the routing profile before any signal
// evaluation. Every virtual entrypoint selects its mapped recipe; concrete
// backend models keep a
// nil recipe so they bypass recipe routing entirely.
func (r *OpenAIRouter) resolveEntrypointForRequest(originalModel string, ctx *RequestContext) {
	if r == nil || r.Config == nil || ctx == nil {
		return
	}
	recipe, ok := r.Config.RecipeForRoutingModel(originalModel)
	if !ok {
		ctx.Routing.SelectPassthrough()
		return
	}
	ctx.Routing.SelectRecipe(recipe)
	observeRoutingIdentity(ctx, originalModel)
	logging.ComponentDebugEvent("extproc", "entrypoint_recipe_resolved", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      originalModel,
		"recipe":     recipe.Name,
	})
}

// resolveIngressEntrypoint finds the entrypoint the request's model names
// before the ingress codec decodes it, because the codec applies that
// entrypoint's accepted losses. The model is read as routing will read it,
// after an unserved model is routed. A hop names a concrete model and keeps
// the entrypoint of the client request it is made for (withHopEntrypoint).
func (r *OpenAIRouter) resolveIngressEntrypoint(body []byte, ctx *RequestContext) {
	if r == nil || r.Config == nil || ctx == nil || ctx.Hop != nil {
		return
	}
	model := gjson.GetBytes(body, "model")
	if model.Type != gjson.String {
		return
	}
	name := r.unservedModelRoute(strings.TrimSpace(model.String()), ctx)
	if entrypoint, ok := r.Config.ResolveEntrypoint(config.ChatAPI, name); ok {
		ctx.entrypoint = &entrypoint
	}
}

type hopEntrypointKey struct{}

// withHopEntrypoint marks the context a Looper or context-recovery hop runs
// under with the entrypoint of the client request reqCtx, so the hop, which
// names a concrete model, takes that request's accepted losses.
func withHopEntrypoint(ctx context.Context, reqCtx *RequestContext) context.Context {
	if reqCtx == nil || reqCtx.entrypoint == nil {
		return ctx
	}
	return context.WithValue(ctx, hopEntrypointKey{}, reqCtx.entrypoint)
}

func hopEntrypointFrom(ctx context.Context) *config.EntrypointMapping {
	if ctx == nil {
		return nil
	}
	entrypoint, _ := ctx.Value(hopEntrypointKey{}).(*config.EntrypointMapping)
	return entrypoint
}

// acceptedLosses are the translation losses accepted for ctx's request: its
// entrypoint's allow_lossy_translation when it sets one, else the global
// list. Every codec and projection for the request reads them here.
func (r *OpenAIRouter) acceptedLosses(ctx *RequestContext) []string {
	if r == nil || r.Config == nil {
		return nil
	}
	var entrypoint *config.EntrypointMapping
	if ctx != nil {
		entrypoint = ctx.entrypoint
	}
	return r.Config.AcceptedLosses(entrypoint)
}

func (r *OpenAIRouter) classifierForRequest(ctx *RequestContext) *classification.Classifier {
	if r == nil || ctx == nil || ctx.Routing.SelectedRecipe() == nil {
		return nil
	}
	recipe := ctx.Routing.SelectedRecipe()
	// Programmatic single-profile routers may provide only the default
	// classifier. Named recipes never fall back across the isolation boundary.
	if r.RecipeClassifiers == nil {
		if r.Config.IsDefaultRecipe(recipe.Name) {
			return r.Classifier
		}
		return nil
	}
	classifier, ok := r.RecipeClassifiers.ForRecipe(recipe.Name)
	if !ok {
		return nil
	}
	return classifier
}

// requestModelActsAsAuto reports whether the inbound model name is resolved by
// the router through the effective entrypoint table rather than forwarded
// as a concrete backend model.
func (r *OpenAIRouter) requestModelActsAsAuto(modelName string) bool {
	if r == nil || r.Config == nil {
		return false
	}
	return r.Config.IsEntrypointModelName(modelName)
}

// decisionCandidatesForRequest scopes evaluation to exactly one recipe.
func (r *OpenAIRouter) decisionCandidatesForRequest(originalModel string, ctx *RequestContext) []config.Decision {
	if ctx != nil && ctx.Routing.SelectedRecipe() != nil {
		recipe := ctx.Routing.SelectedRecipe()
		if recipe.Profile.Decisions == nil {
			// A recipe with no decisions still scopes evaluation: an empty,
			// non-nil slice keeps runDecisionEngine from falling back to the
			// default profile's decisions.
			return []config.Decision{}
		}
		return recipe.Profile.Decisions
	}
	return []config.Decision{}
}
