package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func TestRestrictedListenerListsOnlyItsModels(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/balanced"}},
		Entrypoints:   []config.EntrypointMapping{{ModelNames: []string{"router/flash"}, Recipe: "speed-first"}},
		Recipes:       []config.RoutingRecipe{{Name: "speed-first"}},
	}}
	response, err := router.handleModelsRequest("/v1/models", routing.ListenerModels{"router/flash", "not-in-the-catalog"})
	if err != nil {
		t.Fatal(err)
	}
	var list OpenAIModelList
	if err := json.Unmarshal(response.GetImmediateResponse().Body, &list); err != nil {
		t.Fatal(err)
	}
	if len(list.Data) != 1 || list.Data[0].ID != "router/flash" {
		t.Fatalf("listed %+v, want only router/flash", list.Data)
	}
}

func TestListenerModelsRestrictClientRequestsOnly(t *testing.T) {
	router := &OpenAIRouter{}
	restricted := routing.WithListenerModels(context.Background(), []string{"vllm-sr/auto"})
	client := router.newRoutingSession(restricted, nil)
	if !client.ctx.ListenerModels.Allows("vllm-sr/auto") || client.ctx.ListenerModels.Allows("a") {
		t.Fatalf("client session allow-list = %v", client.ctx.ListenerModels)
	}
	if router.listenerModelRejection("vllm-sr/auto", client.ctx) != nil {
		t.Fatal("an allowed model was rejected")
	}
	rejected := router.listenerModelRejection("a", client.ctx)
	if rejected.GetImmediateResponse().GetStatus().GetCode() != 403 ||
		client.ctx.ImmediateProtocolError == nil || client.ctx.ImmediateProtocolError.Code != "model_not_allowed" {
		t.Fatalf("rejection = %v, protocol error %+v", rejected, client.ctx.ImmediateProtocolError)
	}
	hop := router.newRoutingSession(routing.WithHop(restricted, routing.Hop{Decision: "compare", Iteration: 1}), nil)
	if hop.ctx.ListenerModels != nil || router.listenerModelRejection("a", hop.ctx) != nil {
		t.Fatalf("a request-graph hop took the listener's allow-list %v", hop.ctx.ListenerModels)
	}
}
