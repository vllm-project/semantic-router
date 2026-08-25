package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	stackgateway "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

type dashboardBuilderScenario struct {
	ctx             context.Context
	opts            pkgtestcases.TestCaseOptions
	dashboardURL    string
	dashboardClient *http.Client
	token           string
	namespaceID     string
	userID          string
	gatewayURL      string
	gatewayClient   *http.Client
	targetModel     string
	seed            string
	publicName      string
	sessionID       string
	turnID          string
	approval        dashboardBuilderApproval
	modelIDs        []string
	apiKey          string
	keyID           string
}

func newDashboardBuilderScenario(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	targetModel string,
) (*dashboardBuilderScenario, func(), error) {
	dashboardPort, stopDashboard, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return nil, nil, err
	}
	dashboardURL := fmt.Sprintf("http://localhost:%s", dashboardPort)
	dashboardClient := &http.Client{Timeout: 45 * time.Second}
	token, err := dashboardAuthToken(ctx, dashboardClient, dashboardURL, opts.Verbose)
	if err != nil {
		stopDashboard()
		return nil, nil, fmt.Errorf("authenticate Builder acceptance principal: %w", err)
	}
	namespaceID, userID, err := dashboardAgentIdentity(
		ctx, dashboardClient, dashboardURL, token, opts.Verbose,
	)
	if err != nil {
		stopDashboard()
		return nil, nil, err
	}
	if err := enableDashboardAgentDelegation(
		ctx, dashboardClient, dashboardURL, token, namespaceID, opts.Verbose,
	); err != nil {
		stopDashboard()
		return nil, nil, err
	}

	gatewayOptions := opts
	gatewayOptions.ServiceConfig = stackgateway.DefaultServiceConfig()
	gateway, err := fixtures.OpenServiceSession(ctx, client, gatewayOptions)
	if err != nil {
		stopDashboard()
		return nil, nil, fmt.Errorf("open public Router gateway: %w", err)
	}
	seed := strconv.FormatInt(time.Now().UTC().UnixNano(), 10)
	scenario := &dashboardBuilderScenario{
		ctx:             ctx,
		opts:            opts,
		dashboardURL:    dashboardURL,
		dashboardClient: dashboardClient,
		token:           token,
		namespaceID:     namespaceID,
		userID:          userID,
		gatewayURL:      gateway.BaseURL(),
		gatewayClient:   gateway.HTTPClient(dashboardBuilderDataPlaneTimeout),
		targetModel:     targetModel,
		seed:            seed,
		publicName:      "builder-e2e-" + seed,
	}
	closeScenario := func() {
		gateway.Close()
		stopDashboard()
	}
	return scenario, closeScenario, nil
}

func (scenario *dashboardBuilderScenario) run() error {
	if err := scenario.prepareReview(); err != nil {
		return err
	}
	if err := scenario.prepareAccess(); err != nil {
		return err
	}
	if err := scenario.verifyImmutableConfirmation(); err != nil {
		return err
	}
	if err := scenario.publish(); err != nil {
		return err
	}
	if err := invokeDashboardBuilderModel(
		scenario.ctx, scenario.gatewayClient, scenario.gatewayURL,
		scenario.apiKey, scenario.publicName, false,
	); err != nil {
		return err
	}
	if err := invokeDashboardBuilderModel(
		scenario.ctx, scenario.gatewayClient, scenario.gatewayURL,
		scenario.apiKey, scenario.publicName, true,
	); err != nil {
		return err
	}
	scenario.setDetails()
	return nil
}

func (scenario *dashboardBuilderScenario) prepareReview() error {
	var err error
	scenario.sessionID, scenario.turnID, err = startDashboardBuilderTurn(
		scenario.ctx, scenario.dashboardClient, scenario.dashboardURL,
		scenario.token, scenario.namespaceID, scenario.targetModel,
		scenario.publicName, scenario.seed, scenario.opts.Verbose,
	)
	if err != nil {
		return err
	}
	scenario.approval, err = waitForDashboardBuilderReview(
		scenario.ctx, scenario.dashboardClient, scenario.dashboardURL,
		scenario.token, scenario.namespaceID, scenario.sessionID,
		scenario.turnID, scenario.opts.Verbose,
	)
	if err != nil {
		return err
	}
	events, err := dashboardBuilderEventHistory(
		scenario.ctx, scenario.dashboardClient, scenario.dashboardURL,
		scenario.token, scenario.namespaceID, scenario.sessionID, scenario.opts.Verbose,
	)
	if err != nil {
		return err
	}
	scenario.modelIDs, err = assertDashboardBuilderReview(
		events, scenario.turnID, scenario.publicName, scenario.targetModel, scenario.approval,
	)
	if err != nil {
		return err
	}
	status, err := scenario.turnStatus()
	if err != nil {
		return err
	}
	if status != "waiting_approval" {
		return fmt.Errorf("Builder turn status before human confirmation = %q, want waiting_approval", status)
	}
	return nil
}

func (scenario *dashboardBuilderScenario) prepareAccess() error {
	var err error
	scenario.apiKey, scenario.keyID, err = createDashboardBuilderAccess(
		scenario.ctx, scenario.dashboardClient, scenario.dashboardURL,
		scenario.token, scenario.namespaceID, scenario.userID,
		scenario.approval.Summary.EntrypointID, scenario.modelIDs,
		scenario.seed, scenario.opts.Verbose,
	)
	if err != nil {
		return err
	}
	return waitForDashboardBuilderCredential(
		scenario.ctx, scenario.gatewayClient, scenario.gatewayURL,
		scenario.apiKey, scenario.publicName,
	)
}

func (scenario *dashboardBuilderScenario) verifyImmutableConfirmation() error {
	wrongDigest, err := corruptDashboardBuilderDigest(scenario.approval.PlanDigest)
	if err != nil {
		return err
	}
	if err := commitDashboardBuilderPlan(
		scenario.ctx, scenario.dashboardClient, scenario.dashboardURL,
		scenario.token, scenario.namespaceID, scenario.approval, wrongDigest,
		"dashboard-builder-rejected-"+scenario.seed,
		http.StatusPreconditionFailed, scenario.opts.Verbose,
	); err != nil {
		return fmt.Errorf("verify immutable human confirmation: %w", err)
	}
	if err := assertDashboardBuilderUndiscoverable(
		scenario.ctx, scenario.gatewayClient, scenario.gatewayURL,
		scenario.apiKey, scenario.publicName,
	); err != nil {
		return err
	}
	status, err := scenario.turnStatus()
	if err != nil {
		return err
	}
	if status != "waiting_approval" {
		return fmt.Errorf("Builder turn status after rejected confirmation = %q, want waiting_approval", status)
	}
	return nil
}

func (scenario *dashboardBuilderScenario) publish() error {
	if err := commitDashboardBuilderPlan(
		scenario.ctx, scenario.dashboardClient, scenario.dashboardURL,
		scenario.token, scenario.namespaceID, scenario.approval,
		scenario.approval.PlanDigest, "dashboard-builder-approved-"+scenario.seed,
		http.StatusAccepted, scenario.opts.Verbose,
	); err != nil {
		return fmt.Errorf("commit confirmed Builder publication: %w", err)
	}
	if err := waitForDashboardAgentTarget(
		scenario.ctx, scenario.dashboardClient, scenario.dashboardURL,
		scenario.token, scenario.namespaceID,
		scenario.approval.Summary.EntrypointID, scenario.opts.Verbose,
	); err != nil {
		return err
	}
	if err := waitForDashboardBuilderCompletion(
		scenario.ctx, scenario.dashboardClient, scenario.dashboardURL,
		scenario.token, scenario.namespaceID, scenario.sessionID,
		scenario.turnID, scenario.approval.PlanID, scenario.opts.Verbose,
	); err != nil {
		return err
	}
	return waitForDashboardBuilderDiscovery(
		scenario.ctx, scenario.gatewayClient, scenario.gatewayURL,
		scenario.apiKey, scenario.publicName,
	)
}

func (scenario *dashboardBuilderScenario) turnStatus() (string, error) {
	return dashboardBuilderTurnStatus(
		scenario.ctx, scenario.dashboardClient, scenario.dashboardURL,
		scenario.token, scenario.namespaceID, scenario.sessionID,
		scenario.turnID, scenario.opts.Verbose,
	)
}

func (scenario *dashboardBuilderScenario) setDetails() {
	if scenario.opts.SetDetails == nil {
		return
	}
	scenario.opts.SetDetails(map[string]interface{}{
		"agent_session_id":         scenario.sessionID,
		"agent_turn_id":            scenario.turnID,
		"recipe_id":                scenario.approval.Summary.RecipeID,
		"entrypoint_id":            scenario.approval.Summary.EntrypointID,
		"public_model":             scenario.publicName,
		"api_key_id":               scenario.keyID,
		"assigned_model_count":     len(scenario.modelIDs),
		"human_confirmation":       true,
		"non_stream_invoke_passed": true,
		"stream_invoke_passed":     true,
	})
}
