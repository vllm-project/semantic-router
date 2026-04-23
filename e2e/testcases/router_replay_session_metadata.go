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

func init() {
	pkgtestcases.Register("router-replay-session-list-filter", pkgtestcases.TestCase{
		Description: "GET /v1/router_replay?session_id= returns only replay rows for that session (x-session-id pinned chats)",
		Tags:        []string{"router-replay", "functional", "router-replay-api"},
		Fn:          testRouterReplaySessionListFilter,
	})
	pkgtestcases.Register("router-replay-session-turn-progression", pkgtestcases.TestCase{
		Description: "Two chat completions in the same x-session-id produce increasing turn_index on replay records",
		Tags:        []string{"router-replay", "functional", "router-replay-api"},
		Fn:          testRouterReplaySessionTurnProgression,
	})
}

func testRouterReplaySessionListFilter(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Router replay: session_id list filter")
	}
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open session: %w", err)
	}
	defer session.Close()

	base := fmt.Sprintf("e2e_rs_%d", time.Now().UnixNano())
	sessA := base + "_a"
	sessB := base + "_b"

	if err := postChatCompletionsWithReplayHeaders(ctx, session, postChatOpts{
		sessionID: sessA,
		userID:    "e2e-replay-sess-user",
		content:   "router replay list filter — request A " + base,
	}); err != nil {
		return err
	}
	time.Sleep(2 * time.Second)
	if err := postChatCompletionsWithReplayHeaders(ctx, session, postChatOpts{
		sessionID: sessB,
		userID:    "e2e-replay-sess-user",
		content:   "router replay list filter — request B " + base,
	}); err != nil {
		return err
	}
	time.Sleep(2 * time.Second)

	if err := assertReplayListAllMatchSessionID(session, sessA, opts.Verbose); err != nil {
		return fmt.Errorf("session A filter: %w", err)
	}
	if err := assertReplayListAllMatchSessionID(session, sessB, opts.Verbose); err != nil {
		return fmt.Errorf("session B filter: %w", err)
	}
	return nil
}

func testRouterReplaySessionTurnProgression(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Router replay: turn_index across two turns (same x-session-id)")
	}
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open session: %w", err)
	}
	defer session.Close()

	sess := fmt.Sprintf("e2e_turn_%d", time.Now().UnixNano())
	user := "e2e-replay-turn-user"

	if err := postChatCompletionsWithReplayHeaders(ctx, session, postChatOpts{
		sessionID: sess,
		userID:    user,
		messages: []fixtures.ChatMessage{
			{Role: "user", Content: "turn progression first message " + sess},
		},
	}); err != nil {
		return err
	}
	time.Sleep(2 * time.Second)
	if err := postChatCompletionsWithReplayHeaders(ctx, session, postChatOpts{
		sessionID: sess,
		userID:    user,
		messages: []fixtures.ChatMessage{
			{Role: "user", Content: "turn progression first message " + sess},
			{Role: "assistant", Content: "mock assistant reply for e2e"},
			{Role: "user", Content: "turn progression follow-up " + sess},
		},
	}); err != nil {
		return err
	}
	time.Sleep(2 * time.Second)

	items, err := fetchReplayListForSession(session, sess, 20)
	if err != nil {
		return err
	}
	if len(items) < 2 {
		return fmt.Errorf("expected at least 2 replay rows for session %q, got %d", sess, len(items))
	}

	minTurn, maxTurn := items[0].TurnIndex, items[0].TurnIndex
	for _, it := range items {
		if it.TurnIndex < minTurn {
			minTurn = it.TurnIndex
		}
		if it.TurnIndex > maxTurn {
			maxTurn = it.TurnIndex
		}
	}
	if maxTurn <= minTurn {
		return fmt.Errorf("expected turn_index range for session %q (min=%d max=%d across %d rows)",
			sess, minTurn, maxTurn, len(items))
	}
	if opts.Verbose {
		fmt.Printf("[Test] session %q replay rows=%d turn_index min=%d max=%d\n", sess, len(items), minTurn, maxTurn)
	}
	return nil
}

type postChatOpts struct {
	sessionID string
	userID    string
	content   string
	messages  []fixtures.ChatMessage
}

func postChatCompletionsWithReplayHeaders(ctx context.Context, session *fixtures.ServiceSession, o postChatOpts) error {
	msgs := o.messages
	if len(msgs) == 0 {
		msgs = []fixtures.ChatMessage{{Role: "user", Content: o.content}}
	}
	chat := fixtures.NewChatCompletionsClient(session, 45*time.Second)
	resp, err := chat.Create(ctx, fixtures.ChatCompletionsRequest{
		Model:    "auto",
		User:     o.userID,
		Messages: msgs,
	}, map[string]string{
		"x-authz-user-id": o.userID,
		"x-session-id":    o.sessionID,
	})
	if err != nil {
		return fmt.Errorf("chat completions: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("chat completions status %d: %s", resp.StatusCode, string(resp.Body))
	}
	return nil
}

// replayListItem is a subset of router replay JSON for list endpoints.
type replayListItem struct {
	ID        string `json:"id"`
	SessionID string `json:"session_id"`
	TurnIndex int    `json:"turn_index"`
	Timestamp string `json:"timestamp"`
}

func assertReplayListAllMatchSessionID(session *fixtures.ServiceSession, wantSession string, verbose bool) error {
	items, err := fetchReplayListForSession(session, wantSession, 50)
	if err != nil {
		return err
	}
	if len(items) == 0 {
		return fmt.Errorf("session_id=%q: expected at least one replay row", wantSession)
	}
	for _, it := range items {
		if it.SessionID != wantSession {
			return fmt.Errorf("row %s: session_id=%q, want %q", it.ID, it.SessionID, wantSession)
		}
	}
	if verbose {
		fmt.Printf("[Test] session_id=%q list count=%d\n", wantSession, len(items))
	}
	return nil
}

func fetchReplayListForSession(session *fixtures.ServiceSession, sessionID string, limit int) ([]replayListItem, error) {
	q := url.Values{}
	q.Set("session_id", sessionID)
	q.Set("limit", fmt.Sprintf("%d", limit))
	httpClient := session.HTTPClient(30 * time.Second)
	raw, err := fixtures.DoGETRequest(context.Background(), httpClient, session.BaseURL()+"/v1/router_replay?"+q.Encode())
	if err != nil {
		return nil, fmt.Errorf("GET router_replay list: %w", err)
	}
	if raw.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET router_replay list status %d: %s", raw.StatusCode, string(raw.Body))
	}
	var listResp replayListResponse
	if err := raw.DecodeJSON(&listResp); err != nil {
		return nil, fmt.Errorf("decode list: %w", err)
	}
	var items []replayListItem
	if err := json.Unmarshal(listResp.Data, &items); err != nil {
		return nil, fmt.Errorf("decode list data: %w", err)
	}
	return items, nil
}
