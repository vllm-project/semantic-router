package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const sharedReplayDecisionName = "shared_replay_decision"

func init() {
	pkgtestcases.Register("router-replay-recipe-list-filter", pkgtestcases.TestCase{
		Description: "Recipe filtering isolates named recipes that reuse the same local decision name",
		Tags:        []string{"router-replay", "functional", "router-replay-api", "recipes"},
		Fn:          testRouterReplayRecipeListFilter,
	})
}

func testRouterReplayRecipeListFilter(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open session: %w", err)
	}
	defer session.Close()

	sessionID := fmt.Sprintf("replay_recipe_%d", time.Now().UnixNano())
	recipes := []struct {
		name  string
		model string
	}{
		{name: "replay-first", model: "vllm-sr/replay-first"},
		{name: "replay-second", model: "vllm-sr/replay-second"},
	}
	for _, recipe := range recipes {
		if err := createNamedRecipeReplay(ctx, session, sessionID, recipe.model); err != nil {
			return fmt.Errorf("create replay for recipe %q: %w", recipe.name, err)
		}
	}

	for _, recipe := range recipes {
		rows, err := fetchRecipeReplayRows(ctx, session, sessionID, recipe.name)
		if err != nil {
			return fmt.Errorf("filter recipe %q: %w", recipe.name, err)
		}
		if len(rows) != 1 {
			return fmt.Errorf("recipe %q returned %d replay rows, want 1", recipe.name, len(rows))
		}
		row := rows[0]
		if row.Recipe != recipe.name || row.Decision != sharedReplayDecisionName || row.SessionID != sessionID {
			return fmt.Errorf(
				"recipe %q returned scope recipe=%q decision=%q session_id=%q",
				recipe.name,
				row.Recipe,
				row.Decision,
				row.SessionID,
			)
		}
	}
	return nil
}

func createNamedRecipeReplay(
	ctx context.Context,
	session *fixtures.ServiceSession,
	sessionID string,
	model string,
) error {
	chat := fixtures.NewChatCompletionsClient(session, 45*time.Second)
	response, err := chat.Create(ctx, fixtures.ChatCompletionsRequest{
		Model: model,
		User:  "replay-recipe-user",
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: "record this recipe-scoped replay"},
		},
	}, map[string]string{
		"x-session-id": sessionID,
	})
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("chat completions status %d: %s", response.StatusCode, string(response.Body))
	}
	return nil
}

type recipeReplayListItem struct {
	ID        string `json:"id"`
	Recipe    string `json:"recipe"`
	Decision  string `json:"decision"`
	SessionID string `json:"session_id"`
}

func fetchRecipeReplayRows(
	ctx context.Context,
	session *fixtures.ServiceSession,
	sessionID string,
	recipe string,
) ([]recipeReplayListItem, error) {
	query := url.Values{}
	query.Set("recipe", recipe)
	query.Set("decision", sharedReplayDecisionName)
	query.Set("session_id", sessionID)
	query.Set("limit", "20")

	raw, err := fixtures.DoGETRequest(
		ctx,
		session.HTTPClient(30*time.Second),
		session.BaseURL()+"/v1/router_replay?"+query.Encode(),
	)
	if err != nil {
		return nil, err
	}
	if raw.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET router_replay list status %d: %s", raw.StatusCode, string(raw.Body))
	}

	var listResponse replayListResponse
	if err := raw.DecodeJSON(&listResponse); err != nil {
		return nil, fmt.Errorf("decode list: %w", err)
	}
	var rows []recipeReplayListItem
	if err := json.Unmarshal(listResponse.Data, &rows); err != nil {
		return nil, fmt.Errorf("decode list data: %w", err)
	}
	return rows, nil
}
