package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

func prepareNativeExecutors(cfg *config.RouterConfig) (map[string]*systemone.Executor, error) {
	executors := map[string]*systemone.Executor{}
	for _, recipe := range cfg.ReachableRoutingRecipes() {
		for _, decision := range recipe.Profile.Decisions {
			if !decision.Algorithm.IsNative() {
				continue
			}
			quality, err := systemone.LoadCalibration(cfg, recipe, &decision)
			if err != nil {
				return nil, fmt.Errorf("native recipe %q decision %q calibration: %w", recipe.Name, decision.Name, err)
			}
			executor, err := systemone.NewExecutor(decision.Algorithm, quality)
			if err != nil {
				return nil, fmt.Errorf("native recipe %q decision %q: %w", recipe.Name, decision.Name, err)
			}
			executors[config.RoutingDecisionKey(recipe.Name, decision.Name)] = executor
		}
	}
	return executors, nil
}

// RouteSystemOne shares recipe signal and decision evaluation with Chat while
// preserving the native task and using a separate native algorithm executor.
// invoke is bound to the same retained generation's local and remote backends.
func (r *OpenAIRouter) RouteSystemOne(ctx context.Context, model string, body json.RawMessage, invoke systemone.Invoke) (int, []byte, error) {
	entrypoint, ok := r.Config.ResolveEntrypoint(config.SystemOneAPI, model)
	if !ok {
		return http.StatusNotFound, nil, errors.New("native entrypoint not found")
	}
	recipe, ok := r.Config.RecipeByName(entrypoint.Recipe)
	if !ok {
		return http.StatusServiceUnavailable, nil, errors.New("native recipe unavailable")
	}
	request, err := systemone.ParseNativeRequest(body)
	if err != nil {
		return http.StatusBadRequest, nil, err
	}
	requestContext := &RequestContext{TraceContext: ctx}
	requestContext.Routing.SelectRecipe(recipe)
	classifier := r.classifierForRequest(requestContext)
	if classifier == nil {
		return http.StatusServiceUnavailable, nil, errors.New("native recipe classifier unavailable")
	}
	text := request.SignalText
	signals := classifier.EvaluateAllSignalsWithRequestFactsForDecisions(
		text, text, text, nil, nil, false, false, text, nil,
		classification.ConversationFacts{}, "", classification.RequestFacts{Context: ctx}, recipe.Profile.Decisions,
	)
	if contextErr := ctx.Err(); contextErr != nil {
		return http.StatusGatewayTimeout, nil, contextErr
	}
	decision, err := classifier.EvaluateDecisionWithEngineForDecisions(signals, recipe.Profile.Decisions)
	if err != nil || decision == nil || decision.Decision == nil {
		return http.StatusServiceUnavailable, nil, systemone.ErrUnresolved
	}
	executor := r.nativeExecutors[config.RoutingDecisionKey(recipe.Name, decision.Decision.Name)]
	if executor == nil {
		return http.StatusServiceUnavailable, nil, errors.New("native execution plan unavailable")
	}
	limits := decision.Decision.Algorithm.Budget
	if limits == nil || limits.MaxCalls <= 0 {
		return http.StatusServiceUnavailable, nil, errors.New("native algorithm budget unavailable")
	}
	duration, err := time.ParseDuration(limits.Deadline)
	if err != nil || duration <= 0 {
		return http.StatusServiceUnavailable, nil, errors.New("native algorithm deadline invalid")
	}
	ctx, cancel := context.WithTimeout(ctx, duration)
	defer cancel()
	ctx, ledger := budget.WithLimit(ctx, limits.MaxCalls)
	result, err := executor.Execute(ctx, request, invoke)
	if err != nil {
		if ctx.Err() != nil {
			return http.StatusGatewayTimeout, nil, ctx.Err()
		}
		return http.StatusServiceUnavailable, nil, err
	}
	var response map[string]json.RawMessage
	if json.Unmarshal(result.Body, &response) != nil || response == nil {
		return http.StatusBadGateway, nil, errors.New("invalid native response")
	}
	if !request.ReturnMeta {
		stripSystemOneMetadata(response)
	}
	response["routing"], _ = json.Marshal(map[string]any{
		"recipe": string(recipe.Name), "decision": decision.Decision.Name,
		"algorithm": decision.Decision.Algorithm.Type, "stage": result.Stage, "selected_model": result.Model,
		"quality": decision.Decision.Algorithm.Quality.Type, "model_calls": ledger.Used(),
	})
	encoded, err := json.Marshal(response)
	return http.StatusOK, encoded, err
}

// Metadata visibility applies to each native response envelope. Answers are
// raw JSON: a question, state or answer extension named "meta" remains data.
func stripSystemOneMetadata(response map[string]json.RawMessage) {
	delete(response, "meta")
	var states map[string]json.RawMessage
	if json.Unmarshal(response["states"], &states) != nil || len(states) == 0 {
		return
	}
	for name, raw := range states {
		var state map[string]json.RawMessage
		if json.Unmarshal(raw, &state) != nil {
			continue
		}
		if _, exists := state["meta"]; exists {
			delete(state, "meta")
			states[name], _ = json.Marshal(state)
		}
	}
	response["states"], _ = json.Marshal(states)
}
