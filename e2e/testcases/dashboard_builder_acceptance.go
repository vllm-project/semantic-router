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

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	stackgateway "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
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

	dashboardPort, stopDashboard, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopDashboard()
	dashboardURL := fmt.Sprintf("http://localhost:%s", dashboardPort)
	dashboardClient := &http.Client{Timeout: 45 * time.Second}
	token, err := dashboardAuthToken(ctx, dashboardClient, dashboardURL, opts.Verbose)
	if err != nil {
		return fmt.Errorf("authenticate Builder acceptance principal: %w", err)
	}
	namespaceID, userID, err := dashboardAgentIdentity(
		ctx, dashboardClient, dashboardURL, token, opts.Verbose,
	)
	if err != nil {
		return err
	}
	if err := enableDashboardAgentDelegation(
		ctx, dashboardClient, dashboardURL, token, namespaceID, opts.Verbose,
	); err != nil {
		return err
	}

	gatewayOptions := opts
	gatewayOptions.ServiceConfig = stackgateway.DefaultServiceConfig()
	gatewaySession, err := fixtures.OpenServiceSession(ctx, client, gatewayOptions)
	if err != nil {
		return fmt.Errorf("open public Router gateway: %w", err)
	}
	defer gatewaySession.Close()
	gatewayClient := gatewaySession.HTTPClient(dashboardBuilderDataPlaneTimeout)

	seed := strconv.FormatInt(time.Now().UTC().UnixNano(), 10)
	publicName := "builder-e2e-" + seed
	sessionID, turnID, err := startDashboardBuilderTurn(
		ctx, dashboardClient, dashboardURL, token, namespaceID,
		targetModel, publicName, seed, opts.Verbose,
	)
	if err != nil {
		return err
	}
	approval, err := waitForDashboardBuilderReview(
		ctx, dashboardClient, dashboardURL, token, namespaceID,
		sessionID, turnID, opts.Verbose,
	)
	if err != nil {
		return err
	}
	events, err := dashboardBuilderEventHistory(
		ctx, dashboardClient, dashboardURL, token, namespaceID, sessionID, opts.Verbose,
	)
	if err != nil {
		return err
	}
	modelIDs, err := assertDashboardBuilderReview(events, turnID, publicName, targetModel, approval)
	if err != nil {
		return err
	}
	status, err := dashboardBuilderTurnStatus(
		ctx, dashboardClient, dashboardURL, token, namespaceID, sessionID, turnID, opts.Verbose,
	)
	if err != nil {
		return err
	}
	if status != "waiting_approval" {
		return fmt.Errorf("Builder turn status before human confirmation = %q, want waiting_approval", status)
	}

	apiKey, keyID, err := createDashboardBuilderAccess(
		ctx, dashboardClient, dashboardURL, token, namespaceID, userID,
		approval.Summary.EntrypointID, modelIDs, seed, opts.Verbose,
	)
	if err != nil {
		return err
	}
	if err := waitForDashboardBuilderCredential(
		ctx, gatewayClient, gatewaySession.BaseURL(), apiKey, publicName,
	); err != nil {
		return err
	}

	// Chat text is never publication approval. Prove that the immutable plan
	// digest is required and that a rejected confirmation leaves the draft
	// undiscoverable before submitting the exact review the human saw.
	wrongDigest, err := corruptDashboardBuilderDigest(approval.PlanDigest)
	if err != nil {
		return err
	}
	if err := commitDashboardBuilderPlan(
		ctx, dashboardClient, dashboardURL, token, namespaceID, approval,
		wrongDigest, "dashboard-builder-rejected-"+seed,
		http.StatusPreconditionFailed, opts.Verbose,
	); err != nil {
		return fmt.Errorf("verify immutable human confirmation: %w", err)
	}
	if err := assertDashboardBuilderUndiscoverable(
		ctx, gatewayClient, gatewaySession.BaseURL(), apiKey, publicName,
	); err != nil {
		return err
	}
	status, err = dashboardBuilderTurnStatus(
		ctx, dashboardClient, dashboardURL, token, namespaceID, sessionID, turnID, opts.Verbose,
	)
	if err != nil {
		return err
	}
	if status != "waiting_approval" {
		return fmt.Errorf("Builder turn status after rejected confirmation = %q, want waiting_approval", status)
	}
	if err := commitDashboardBuilderPlan(
		ctx, dashboardClient, dashboardURL, token, namespaceID, approval,
		approval.PlanDigest, "dashboard-builder-approved-"+seed,
		http.StatusAccepted, opts.Verbose,
	); err != nil {
		return fmt.Errorf("commit confirmed Builder publication: %w", err)
	}
	if err := waitForDashboardAgentTarget(
		ctx, dashboardClient, dashboardURL, token, namespaceID,
		approval.Summary.EntrypointID, opts.Verbose,
	); err != nil {
		return err
	}
	if err := waitForDashboardBuilderCompletion(
		ctx, dashboardClient, dashboardURL, token, namespaceID,
		sessionID, turnID, approval.PlanID, opts.Verbose,
	); err != nil {
		return err
	}
	if err := waitForDashboardBuilderDiscovery(
		ctx, gatewayClient, gatewaySession.BaseURL(), apiKey, publicName,
	); err != nil {
		return err
	}
	if err := invokeDashboardBuilderModel(
		ctx, gatewayClient, gatewaySession.BaseURL(), apiKey, publicName, false,
	); err != nil {
		return err
	}
	if err := invokeDashboardBuilderModel(
		ctx, gatewayClient, gatewaySession.BaseURL(), apiKey, publicName, true,
	); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"agent_session_id":         sessionID,
			"agent_turn_id":            turnID,
			"recipe_id":                approval.Summary.RecipeID,
			"entrypoint_id":            approval.Summary.EntrypointID,
			"public_model":             publicName,
			"api_key_id":               keyID,
			"assigned_model_count":     len(modelIDs),
			"human_confirmation":       true,
			"non_stream_invoke_passed": true,
			"stream_invoke_passed":     true,
		})
	}
	return nil
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
		page, err := dashboardBuilderEventsPage(
			ctx, client, baseURL, token, namespaceID, sessionID, "", verbose,
		)
		if err != nil {
			lastPhase = err.Error()
		} else {
			for _, event := range page.Data {
				if event.TurnID != turnID {
					continue
				}
				lastPhase = event.Type
				switch event.Type {
				case "approval_request":
					var approval dashboardBuilderApproval
					if err := json.Unmarshal(event.Payload, &approval); err != nil {
						return dashboardBuilderApproval{}, fmt.Errorf("decode Builder publication review: %w", err)
					}
					return approval, nil
				case "terminal":
					var terminal struct {
						Status string `json:"status"`
						Error  *struct {
							Code    string `json:"code"`
							Message string `json:"message"`
						} `json:"error"`
					}
					if err := json.Unmarshal(event.Payload, &terminal); err != nil {
						return dashboardBuilderApproval{}, fmt.Errorf("decode Builder terminal event: %w", err)
					}
					return dashboardBuilderApproval{}, fmt.Errorf(
						"Builder turn terminated before publication review (status=%q error=%v)",
						terminal.Status, terminal.Error,
					)
				}
			}
		}
		timer := time.NewTimer(2 * time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			return dashboardBuilderApproval{}, ctx.Err()
		case <-timer.C:
		}
	}
	return dashboardBuilderApproval{}, fmt.Errorf(
		"Builder did not prepare a publication review within %s (last phase: %s)",
		dashboardBuilderTurnTimeout, lastPhase,
	)
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

func assertDashboardBuilderReview(
	events []dashboardBuilderEvent,
	turnID string,
	publicName string,
	targetModel string,
	approval dashboardBuilderApproval,
) ([]string, error) {
	if approval.PlanID == "" || approval.PlanDigest == "" || approval.PlanRevision < 1 ||
		approval.PlanETag == "" || !time.Now().UTC().Before(approval.ExpiresAt.UTC()) {
		return nil, fmt.Errorf("Builder publication review omitted immutable confirmation metadata")
	}
	if approval.Summary.RecipeID == "" || approval.Summary.RecipeName == "" ||
		approval.Summary.EntrypointID == "" || approval.Summary.EntrypointName != publicName {
		return nil, fmt.Errorf("Builder publication review does not describe the requested Recipe and Entrypoint")
	}

	completed := make(map[string]int, len(dashboardBuilderRequiredTools))
	validationPassed := false
	probePassed := false
	evaluationPassed := false
	targetModelIDs := make(map[string]struct{})
	createInvocations := make(map[string]string)
	createdRecipeIDs := make(map[string]struct{})
	createdEntrypointIDs := make(map[string]struct{})
	for _, event := range events {
		if event.TurnID != turnID {
			continue
		}
		if event.Type == "tool_request" {
			var request struct {
				InvocationID string          `json:"invocationId"`
				ToolName     string          `json:"toolName"`
				Arguments    json.RawMessage `json:"arguments"`
			}
			if err := json.Unmarshal(event.Payload, &request); err != nil {
				return nil, fmt.Errorf("decode Builder tool request: %w", err)
			}
			if request.ToolName == "router.recipe.prepare" || request.ToolName == "router.entrypoint.prepare" {
				var arguments struct {
					ExpectedRevision int64 `json:"expectedRevision"`
				}
				if err := json.Unmarshal(request.Arguments, &arguments); err != nil {
					return nil, fmt.Errorf("decode Builder draft request: %w", err)
				}
				if arguments.ExpectedRevision == 0 {
					createInvocations[request.InvocationID] = request.ToolName
				}
			}
			continue
		}
		if event.Type != "tool_result" {
			continue
		}
		var result struct {
			InvocationID string          `json:"invocationId"`
			ToolName     string          `json:"toolName"`
			Status       string          `json:"status"`
			Result       json.RawMessage `json:"result"`
			Error        *struct {
				Code    string `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(event.Payload, &result); err != nil {
			return nil, fmt.Errorf("decode Builder tool result: %w", err)
		}
		if result.Status != "completed" || result.Error != nil {
			return nil, fmt.Errorf(
				"Builder tool %q did not complete cleanly (status=%q error=%v)",
				result.ToolName, result.Status, result.Error,
			)
		}
		completed[result.ToolName]++
		switch result.ToolName {
		case "router.recipe.prepare":
			var value struct {
				RecipeID string `json:"recipeId"`
			}
			if err := json.Unmarshal(result.Result, &value); err != nil {
				return nil, fmt.Errorf("decode prepared Recipe: %w", err)
			}
			if createInvocations[result.InvocationID] == result.ToolName && value.RecipeID != "" {
				createdRecipeIDs[value.RecipeID] = struct{}{}
			}
		case "router.entrypoint.prepare":
			var value struct {
				EntrypointID string `json:"entrypointId"`
			}
			if err := json.Unmarshal(result.Result, &value); err != nil {
				return nil, fmt.Errorf("decode prepared Entrypoint: %w", err)
			}
			if createInvocations[result.InvocationID] == result.ToolName && value.EntrypointID != "" {
				createdEntrypointIDs[value.EntrypointID] = struct{}{}
			}
		case "router.models.list":
			var value struct {
				Data []struct {
					ID     string `json:"id"`
					Name   string `json:"name"`
					Status string `json:"status"`
					Card   struct {
						Aliases []string `json:"aliases"`
					} `json:"card"`
				} `json:"data"`
			}
			if err := json.Unmarshal(result.Result, &value); err != nil {
				return nil, fmt.Errorf("decode connected Model discovery: %w", err)
			}
			for _, model := range value.Data {
				if model.Status == "active" && (model.ID == targetModel || model.Name == targetModel ||
					dashboardBuilderContains(model.Card.Aliases, targetModel)) {
					targetModelIDs[model.ID] = struct{}{}
				}
			}
		case "router.recipe.validate":
			var value struct {
				Valid bool `json:"valid"`
			}
			if err := json.Unmarshal(result.Result, &value); err != nil {
				return nil, fmt.Errorf("decode Recipe validation evidence: %w", err)
			}
			validationPassed = validationPassed || value.Valid
		case "router.recipe.probe":
			var value struct {
				Passed bool `json:"passed"`
			}
			if err := json.Unmarshal(result.Result, &value); err != nil {
				return nil, fmt.Errorf("decode Recipe probe evidence: %w", err)
			}
			probePassed = probePassed || value.Passed
		case "router.recipe.evaluate":
			var value struct {
				Passed bool `json:"passed"`
			}
			if err := json.Unmarshal(result.Result, &value); err != nil {
				return nil, fmt.Errorf("decode Recipe evaluation evidence: %w", err)
			}
			evaluationPassed = evaluationPassed || value.Passed
		}
	}
	for _, toolName := range dashboardBuilderRequiredTools {
		if completed[toolName] == 0 {
			return nil, fmt.Errorf("Builder publication review is missing completed tool %q", toolName)
		}
	}
	if !validationPassed || !probePassed || !evaluationPassed {
		return nil, fmt.Errorf(
			"Builder evidence did not pass (validation=%t probe=%t evaluation=%t)",
			validationPassed, probePassed, evaluationPassed,
		)
	}
	if _, created := createdRecipeIDs[approval.Summary.RecipeID]; !created {
		return nil, fmt.Errorf("Builder publication review did not use the Recipe created in this turn")
	}
	if _, created := createdEntrypointIDs[approval.Summary.EntrypointID]; !created {
		return nil, fmt.Errorf("Builder publication review did not use the Entrypoint created in this turn")
	}
	var gates []struct {
		Name   string `json:"name"`
		Passed bool   `json:"passed"`
	}
	if err := json.Unmarshal(approval.Summary.GateResults, &gates); err != nil {
		return nil, fmt.Errorf("decode Builder publication gate results: %w", err)
	}
	if len(gates) == 0 {
		return nil, fmt.Errorf("Builder publication review omitted gate results")
	}
	for _, gate := range gates {
		if gate.Name == "" || !gate.Passed {
			return nil, fmt.Errorf("Builder publication gate did not pass: %+v", gate)
		}
	}
	modelIDs, err := dashboardBuilderAssignmentModelIDs(approval.Summary.Assignments)
	if err != nil {
		return nil, err
	}
	if len(modelIDs) == 0 {
		return nil, fmt.Errorf("Builder publication review has no assigned Models")
	}
	if len(targetModelIDs) == 0 {
		return nil, fmt.Errorf("Builder did not discover the real session target %q", targetModel)
	}
	for _, modelID := range modelIDs {
		if _, expected := targetModelIDs[modelID]; !expected {
			return nil, fmt.Errorf(
				"Builder assigned Model %q instead of the real session target %q",
				modelID, targetModel,
			)
		}
	}
	return modelIDs, nil
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
			if approvalCommitted && terminalCompleted {
				status, statusErr := dashboardBuilderTurnStatus(
					ctx, client, baseURL, token, namespaceID, sessionID, turnID, verbose,
				)
				if statusErr == nil && status == "completed" {
					return nil
				}
			}
		}
		timer := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return fmt.Errorf("Builder confirmation did not produce committed approval and completed turn events")
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
