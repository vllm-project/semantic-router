package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const (
	dashboardBuilderModelEnvironment = "VLLM_SR_E2E_BUILDER_MODEL"
	dashboardBuilderTurnTimeout      = 14 * time.Minute
	dashboardBuilderDataPlaneTimeout = 3 * time.Minute
	dashboardBuilderEventPageSize    = 200
)

var dashboardBuilderRequiredTools = []string{
	"router.catalog.describe",
	"router.skills.read",
	"router.models.list",
	"router.recipes.examples",
	"router.recipe.prepare",
	"router.recipe.validate",
	"router.entrypoint.prepare",
	"router.recipe.probe",
	"router.recipe.evaluate",
	"router.publish.prepare",
}

type dashboardBuilderEvent struct {
	TurnID   string          `json:"turnId"`
	Sequence int64           `json:"sequence"`
	Type     string          `json:"type"`
	Payload  json.RawMessage `json:"payload"`
}

type dashboardBuilderEventPage struct {
	Data []dashboardBuilderEvent `json:"data"`
	Page struct {
		NextCursor string `json:"nextCursor"`
		HasMore    bool   `json:"hasMore"`
	} `json:"page"`
}

type dashboardBuilderApproval struct {
	PlanID       string    `json:"planId"`
	PlanDigest   string    `json:"planDigest"`
	PlanRevision int64     `json:"planRevision"`
	PlanETag     string    `json:"planEtag"`
	ExpiresAt    time.Time `json:"expiresAt"`
	Summary      struct {
		RecipeID       string          `json:"recipeId"`
		RecipeName     string          `json:"recipeName"`
		EntrypointID   string          `json:"entrypointId"`
		EntrypointName string          `json:"entrypointName"`
		Assignments    json.RawMessage `json:"assignments"`
		GateResults    json.RawMessage `json:"gateResults"`
	} `json:"summary"`
}

func init() {
	// Keep this real-model acceptance opt-in. The default dashboard profile uses
	// deterministic simulated inference for its ordinary CI contract, which is
	// intentionally insufficient to prove autonomous Builder tool use.
	pkgtestcases.Register("dashboard-builder-publication", pkgtestcases.TestCase{
		Description: "A real Router Agent designs, validates, probes, evaluates, confirms, publishes, and invokes a Mixture-of-Models",
		Tags:        []string{"dashboard", "agent", "builder", "real-model", "publication"},
		Fn:          testDashboardBuilderPublication,
	})
}

func testDashboardBuilderPublication(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	if opts.Profile != "dashboard" {
		return fmt.Errorf("dashboard-builder-publication requires the dashboard E2E profile")
	}
	targetModel := strings.TrimSpace(os.Getenv(dashboardBuilderModelEnvironment))
	if targetModel == "" {
		return fmt.Errorf(
			"%s must name an authorized, connected, tool-capable real Model; simulated inference does not satisfy this acceptance",
			dashboardBuilderModelEnvironment,
		)
	}
	if opts.Verbose {
		fmt.Println("[Test] Testing the complete Router-native Builder publication flow")
	}

	scenario, closeScenario, err := newDashboardBuilderScenario(ctx, client, opts, targetModel)
	if err != nil {
		return err
	}
	defer closeScenario()
	return scenario.run()
}

func startDashboardBuilderTurn(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	targetModel string,
	publicName string,
	seed string,
	verbose bool,
) (string, string, error) {
	sessionReceipt, err := dashboardManagementMutation(
		ctx, client, baseURL, token, namespaceID,
		http.MethodPost, "/agent-sessions", "dashboard-builder-session-"+seed,
		map[string]interface{}{
			"mode": "builder",
			"target": map[string]string{
				"kind": "model",
				"id":   targetModel,
			},
			"title": "Builder publication acceptance",
		},
		verbose,
	)
	if err != nil {
		return "", "", fmt.Errorf(
			"create Builder session (ensure %s is visible to the linked User): %w",
			dashboardBuilderModelEnvironment, err,
		)
	}
	if sessionReceipt.Resource == nil || sessionReceipt.Resource.ID == "" {
		return "", "", fmt.Errorf("create Builder session returned no resource")
	}

	prompt := fmt.Sprintf(`Build a new production-ready, model-free Recipe and one Entrypoint whose name and only public alias are exactly %q.

Complete the full Router-native Builder workflow in this turn. First load the pinned Builder skill. Inspect the live component catalog, model-free Recipe examples, and authorized connected Models through their tools; do not guess schemas or reuse an existing Recipe or Entrypoint. Search the Model catalog for the real session target %q and assign only that Model to every Decision. Design a concise Recipe with at least one Decision, create its draft, validate the persisted draft, and create the Entrypoint draft. Run the connectivity probe and readiness evaluation against those exact revisions. Only if validation, probes, and every evaluation gate pass, prepare an immutable publication review. Stop at that review and wait for the separate Management confirmation; do not claim that chat text published anything.`, publicName, targetModel)
	turnReceipt, err := dashboardManagementMutation(
		ctx, client, baseURL, token, namespaceID,
		http.MethodPost, "/agent-sessions/"+sessionReceipt.Resource.ID+"/turns",
		"dashboard-builder-turn-"+seed,
		map[string]interface{}{
			"input": map[string]interface{}{
				"content": []map[string]string{{"type": "text", "text": prompt}},
			},
		},
		verbose,
	)
	if err != nil {
		return "", "", fmt.Errorf("create Builder turn: %w", err)
	}
	if turnReceipt.Resource == nil || turnReceipt.Resource.ID == "" {
		return "", "", fmt.Errorf("create Builder turn returned no resource")
	}
	return sessionReceipt.Resource.ID, turnReceipt.Resource.ID, nil
}

func waitForDashboardBuilderReview(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	sessionID string,
	turnID string,
	verbose bool,
) (dashboardBuilderApproval, error) {
	deadline := time.Now().Add(dashboardBuilderTurnTimeout)
	lastPhase := "no events"
	for time.Now().Before(deadline) {
		approval, phase, ready, err := pollDashboardBuilderReview(
			ctx, client, baseURL, token, namespaceID, sessionID, turnID, "", verbose,
		)
		if phase != "" {
			lastPhase = phase
		}
		if err != nil {
			return dashboardBuilderApproval{}, err
		}
		if ready {
			return approval, nil
		}
		if err := waitDashboardBuilderPoll(ctx, 2*time.Second); err != nil {
			return dashboardBuilderApproval{}, err
		}
	}
	return dashboardBuilderApproval{}, fmt.Errorf(
		"Builder did not prepare a publication review within %s (last phase: %s)",
		dashboardBuilderTurnTimeout, lastPhase,
	)
}

func pollDashboardBuilderReview(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	sessionID string,
	turnID string,
	cursor string,
	verbose bool,
) (dashboardBuilderApproval, string, bool, error) {
	page, err := dashboardBuilderEventsPage(
		ctx, client, baseURL, token, namespaceID, sessionID, cursor, verbose,
	)
	if err != nil {
		return dashboardBuilderApproval{}, err.Error(), false, nil
	}
	return inspectDashboardBuilderReviewEvents(page.Data, turnID)
}

func inspectDashboardBuilderReviewEvents(
	events []dashboardBuilderEvent,
	turnID string,
) (dashboardBuilderApproval, string, bool, error) {
	lastPhase := ""
	for _, event := range events {
		if event.TurnID != turnID {
			continue
		}
		lastPhase = event.Type
		switch event.Type {
		case "approval_request":
			var approval dashboardBuilderApproval
			if err := json.Unmarshal(event.Payload, &approval); err != nil {
				return dashboardBuilderApproval{}, lastPhase, false,
					fmt.Errorf("decode Builder publication review: %w", err)
			}
			return approval, lastPhase, true, nil
		case "terminal":
			var terminal struct {
				Status string `json:"status"`
				Error  *struct {
					Code    string `json:"code"`
					Message string `json:"message"`
				} `json:"error"`
			}
			if err := json.Unmarshal(event.Payload, &terminal); err != nil {
				return dashboardBuilderApproval{}, lastPhase, false,
					fmt.Errorf("decode Builder terminal event: %w", err)
			}
			return dashboardBuilderApproval{}, lastPhase, false, fmt.Errorf(
				"Builder turn terminated before publication review (status=%q error=%v)",
				terminal.Status, terminal.Error,
			)
		}
	}
	return dashboardBuilderApproval{}, lastPhase, false, nil
}

func waitDashboardBuilderPoll(ctx context.Context, interval time.Duration) error {
	timer := time.NewTimer(interval)
	select {
	case <-ctx.Done():
		timer.Stop()
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func dashboardBuilderEventHistory(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	sessionID string,
	verbose bool,
) ([]dashboardBuilderEvent, error) {
	cursor := ""
	events := make([]dashboardBuilderEvent, 0, dashboardBuilderEventPageSize)
	for pageNumber := 0; pageNumber < 32; pageNumber++ {
		page, err := dashboardBuilderEventsPage(
			ctx, client, baseURL, token, namespaceID, sessionID, cursor, verbose,
		)
		if err != nil {
			return nil, err
		}
		events = append(events, page.Data...)
		if !page.Page.HasMore {
			sort.Slice(events, func(left, right int) bool {
				return events[left].Sequence < events[right].Sequence
			})
			for index := 1; index < len(events); index++ {
				if events[index].Sequence <= events[index-1].Sequence {
					return nil, fmt.Errorf("Builder event history contains duplicate or unordered sequences")
				}
			}
			return events, nil
		}
		if page.Page.NextCursor == "" || page.Page.NextCursor == cursor {
			return nil, fmt.Errorf("Builder event history returned an invalid pagination cursor")
		}
		cursor = page.Page.NextCursor
	}
	return nil, fmt.Errorf("Builder event history exceeded the bounded acceptance transcript")
}

func dashboardBuilderEventsPage(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	sessionID string,
	cursor string,
	verbose bool,
) (dashboardBuilderEventPage, error) {
	path := "/agent-sessions/" + sessionID + "/events?pageSize=" + strconv.Itoa(dashboardBuilderEventPageSize)
	if cursor != "" {
		path += "&cursor=" + url.QueryEscape(cursor)
	}
	var page dashboardBuilderEventPage
	if err := dashboardManagementJSON(
		ctx, client, baseURL, token, namespaceID, http.MethodGet,
		path, "", nil, http.StatusOK, &page, verbose,
	); err != nil {
		return dashboardBuilderEventPage{}, fmt.Errorf("read Builder event history: %w", err)
	}
	return page, nil
}

func dashboardBuilderAssignmentModelIDs(raw json.RawMessage) ([]string, error) {
	var document interface{}
	if err := json.Unmarshal(raw, &document); err != nil {
		return nil, fmt.Errorf("decode Builder assignment review: %w", err)
	}
	models := make(map[string]struct{})
	var walk func(interface{})
	walk = func(value interface{}) {
		switch typed := value.(type) {
		case map[string]interface{}:
			for key, child := range typed {
				if key == "modelId" {
					if modelID, ok := child.(string); ok && modelID != "" {
						models[modelID] = struct{}{}
					}
					continue
				}
				walk(child)
			}
		case []interface{}:
			for _, child := range typed {
				walk(child)
			}
		}
	}
	walk(document)
	result := make([]string, 0, len(models))
	for modelID := range models {
		result = append(result, modelID)
	}
	sort.Strings(result)
	return result, nil
}

func dashboardBuilderTurnStatus(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	sessionID string,
	turnID string,
	verbose bool,
) (string, error) {
	var page struct {
		Data []struct {
			ID     string `json:"id"`
			Status string `json:"status"`
		} `json:"data"`
	}
	if err := dashboardManagementJSON(
		ctx, client, baseURL, token, namespaceID, http.MethodGet,
		"/agent-sessions/"+sessionID+"/turns?pageSize=200", "", nil,
		http.StatusOK, &page, verbose,
	); err != nil {
		return "", fmt.Errorf("read Builder turn: %w", err)
	}
	for _, turn := range page.Data {
		if turn.ID == turnID {
			return turn.Status, nil
		}
	}
	return "", fmt.Errorf("Builder turn %s is missing", turnID)
}

func createDashboardBuilderAccess(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	userID string,
	entrypointID string,
	modelIDs []string,
	seed string,
	verbose bool,
) (string, string, error) {
	grants := []map[string]string{
		{"resourceType": "entrypoint", "resourceId": entrypointID, "permission": "discover", "effect": "allow"},
		{"resourceType": "entrypoint", "resourceId": entrypointID, "permission": "invoke", "effect": "allow"},
	}
	for _, modelID := range modelIDs {
		grants = append(grants,
			map[string]string{"resourceType": "model", "resourceId": modelID, "permission": "discover", "effect": "allow"},
			map[string]string{"resourceType": "model", "resourceId": modelID, "permission": "invoke", "effect": "allow"},
		)
	}
	policyReceipt, err := dashboardManagementMutation(
		ctx, client, baseURL, token, namespaceID,
		http.MethodPost, "/access-policies", "dashboard-builder-policy-"+seed,
		map[string]interface{}{
			"name":   "Builder acceptance " + seed,
			"status": "active",
			"grants": grants,
		},
		verbose,
	)
	if err != nil || policyReceipt.Resource == nil || policyReceipt.Resource.ID == "" {
		return "", "", fmt.Errorf(
			"create Builder acceptance Access Policy: %w",
			fixtureReceiptError(err, policyReceipt.Resource),
		)
	}
	var issued struct {
		Data struct {
			KeyID string `json:"keyId"`
		} `json:"data"`
		Secret string `json:"secret"`
	}
	if err := dashboardManagementJSON(
		ctx, client, baseURL, token, namespaceID, http.MethodPost, "/api-keys",
		"dashboard-builder-key-"+seed,
		map[string]interface{}{
			"name": "Builder acceptance " + seed,
			"owner": map[string]string{
				"type": "user",
				"id":   userID,
			},
			"revealable":      false,
			"accessPolicyIds": []string{policyReceipt.Resource.ID},
		},
		http.StatusCreated, &issued, verbose,
	); err != nil {
		return "", "", fmt.Errorf("create Builder acceptance API key: %w", err)
	}
	if issued.Data.KeyID == "" || issued.Secret == "" {
		return "", "", fmt.Errorf("create Builder acceptance API key omitted its identity or one-time secret")
	}
	return issued.Secret, issued.Data.KeyID, nil
}

func waitForDashboardBuilderCredential(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	apiKey string,
	forbiddenModel string,
) error {
	deadline := time.Now().Add(90 * time.Second)
	var lastErr error
	for time.Now().Before(deadline) {
		models, err := dashboardBuilderAuthorizedModels(ctx, client, baseURL, apiKey)
		if err == nil {
			if dashboardBuilderContains(models, forbiddenModel) {
				return fmt.Errorf("draft Builder Entrypoint %q was discoverable before confirmation", forbiddenModel)
			}
			return nil
		}
		lastErr = err
		timer := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return fmt.Errorf("Builder acceptance API key did not become usable: %w", lastErr)
}

func assertDashboardBuilderUndiscoverable(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	apiKey string,
	model string,
) error {
	models, err := dashboardBuilderAuthorizedModels(ctx, client, baseURL, apiKey)
	if err != nil {
		return err
	}
	if dashboardBuilderContains(models, model) {
		return fmt.Errorf("Builder Entrypoint %q became discoverable after rejected confirmation", model)
	}
	return nil
}

func commitDashboardBuilderPlan(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	approval dashboardBuilderApproval,
	digest string,
	idempotencyKey string,
	expectedStatus int,
	verbose bool,
) error {
	_, err := dashboardManagementJSONWithHeaders(
		ctx, client, baseURL, token, namespaceID, http.MethodPost,
		"/publication-plans/"+approval.PlanID+":commit", idempotencyKey,
		map[string]string{"planDigest": digest}, expectedStatus, nil,
		http.Header{"If-Match": []string{approval.PlanETag}}, verbose,
	)
	return err
}

func corruptDashboardBuilderDigest(digest string) (string, error) {
	if len(digest) < 2 {
		return "", fmt.Errorf("Builder publication review returned an invalid digest")
	}
	last := digest[len(digest)-1]
	replacement := byte('0')
	if last == replacement {
		replacement = '1'
	}
	return digest[:len(digest)-1] + string(replacement), nil
}

func waitForDashboardBuilderCompletion(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	sessionID string,
	turnID string,
	planID string,
	verbose bool,
) error {
	deadline := time.Now().Add(90 * time.Second)
	for time.Now().Before(deadline) {
		events, err := dashboardBuilderEventHistory(
			ctx, client, baseURL, token, namespaceID, sessionID, verbose,
		)
		if err == nil {
			approvalCommitted, terminalCompleted := dashboardBuilderCompletionEvents(
				events, turnID, planID,
			)
			if approvalCommitted && terminalCompleted {
				status, statusErr := dashboardBuilderTurnStatus(
					ctx, client, baseURL, token, namespaceID, sessionID, turnID, verbose,
				)
				if statusErr == nil && status == "completed" {
					return nil
				}
			}
		}
		if err := waitDashboardBuilderPoll(ctx, time.Second); err != nil {
			return err
		}
	}
	return fmt.Errorf("Builder confirmation did not produce committed approval and completed turn events")
}

func dashboardBuilderCompletionEvents(
	events []dashboardBuilderEvent,
	turnID string,
	planID string,
) (bool, bool) {
	approvalCommitted := false
	terminalCompleted := false
	for _, event := range events {
		if event.TurnID != turnID {
			continue
		}
		switch event.Type {
		case "approval_result":
			var result struct {
				PlanID string `json:"planId"`
				Status string `json:"status"`
			}
			if json.Unmarshal(event.Payload, &result) == nil {
				approvalCommitted = result.PlanID == planID && result.Status == "committed"
			}
		case "terminal":
			var terminal struct {
				Status string `json:"status"`
			}
			if json.Unmarshal(event.Payload, &terminal) == nil {
				terminalCompleted = terminal.Status == "completed"
			}
		}
	}
	return approvalCommitted, terminalCompleted
}

func waitForDashboardBuilderDiscovery(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	apiKey string,
	model string,
) error {
	deadline := time.Now().Add(90 * time.Second)
	var lastErr error
	for time.Now().Before(deadline) {
		models, err := dashboardBuilderAuthorizedModels(ctx, client, baseURL, apiKey)
		if err == nil && dashboardBuilderContains(models, model) {
			return nil
		}
		if err != nil {
			lastErr = err
		} else {
			lastErr = fmt.Errorf("authorized Models do not contain %q", model)
		}
		timer := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return fmt.Errorf("published Builder Entrypoint was not discoverable: %w", lastErr)
}

func dashboardBuilderAuthorizedModels(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	apiKey string,
) ([]string, error) {
	req, err := http.NewRequestWithContext(
		ctx, http.MethodGet, strings.TrimRight(baseURL, "/")+"/v1/models", nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create authorized Models request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("list authorized Models: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("read authorized Models: %w", err)
	}
	defer clear(body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf(
			"list authorized Models: expected 200, got %d: %s",
			resp.StatusCode, truncateString(string(body), 240),
		)
	}
	var response openAIModelsResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("decode authorized Models: %w", err)
	}
	if response.Object != "list" {
		return nil, fmt.Errorf("authorized Models object = %q, want list", response.Object)
	}
	models := make([]string, 0, len(response.Data))
	for _, model := range response.Data {
		models = append(models, model.ID)
	}
	return models, nil
}

func dashboardBuilderContains(values []string, expected string) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}

func invokeDashboardBuilderModel(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	apiKey string,
	model string,
	stream bool,
) error {
	payload, err := json.Marshal(map[string]interface{}{
		"model": model,
		"messages": []map[string]string{{
			"role":    "user",
			"content": "Reply with one short sentence confirming the Builder acceptance request.",
		}},
		"max_tokens": 64,
		"stream":     stream,
	})
	if err != nil {
		return fmt.Errorf("marshal Builder invocation: %w", err)
	}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost,
		strings.TrimRight(baseURL, "/")+"/v1/chat/completions", bytes.NewReader(payload),
	)
	if err != nil {
		return fmt.Errorf("create Builder invocation: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("invoke published Builder Entrypoint (stream=%t): %w", stream, err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return fmt.Errorf("read published Builder response (stream=%t): %w", stream, err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf(
			"published Builder invocation (stream=%t): expected 200, got %d: %s",
			stream, resp.StatusCode, truncateString(string(body), 300),
		)
	}
	if stream {
		if !strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") ||
			!bytes.Contains(body, []byte("data:")) || !bytes.Contains(body, []byte("[DONE]")) {
			return fmt.Errorf("published Builder streaming response is not a valid event stream")
		}
		return nil
	}
	var response struct {
		Choices []json.RawMessage `json:"choices"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return fmt.Errorf("decode published Builder response: %w", err)
	}
	if len(response.Choices) == 0 {
		return fmt.Errorf("published Builder response has no choices")
	}
	return nil
}
