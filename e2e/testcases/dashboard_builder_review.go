package testcases

import (
	"encoding/json"
	"fmt"
	"time"
)

type dashboardBuilderToolRequest struct {
	InvocationID string          `json:"invocationId"`
	ToolName     string          `json:"toolName"`
	Arguments    json.RawMessage `json:"arguments"`
}

type dashboardBuilderToolResult struct {
	InvocationID string          `json:"invocationId"`
	ToolName     string          `json:"toolName"`
	Status       string          `json:"status"`
	Result       json.RawMessage `json:"result"`
	Error        *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

type dashboardBuilderReviewEvidence struct {
	completed            map[string]int
	validationPassed     bool
	probePassed          bool
	evaluationPassed     bool
	targetModelIDs       map[string]struct{}
	createInvocations    map[string]string
	createdRecipeIDs     map[string]struct{}
	createdEntrypointIDs map[string]struct{}
}

func assertDashboardBuilderReview(
	events []dashboardBuilderEvent,
	turnID string,
	publicName string,
	targetModel string,
	approval dashboardBuilderApproval,
) ([]string, error) {
	if err := assertDashboardBuilderApprovalMetadata(approval, publicName); err != nil {
		return nil, err
	}
	evidence, err := collectDashboardBuilderReviewEvidence(events, turnID, targetModel)
	if err != nil {
		return nil, err
	}
	if err := evidence.assertRequired(approval); err != nil {
		return nil, err
	}
	if err := assertDashboardBuilderGates(approval.Summary.GateResults); err != nil {
		return nil, err
	}
	modelIDs, err := dashboardBuilderAssignmentModelIDs(approval.Summary.Assignments)
	if err != nil {
		return nil, err
	}
	if err := evidence.assertAssignments(modelIDs, targetModel); err != nil {
		return nil, err
	}
	return modelIDs, nil
}

func assertDashboardBuilderApprovalMetadata(
	approval dashboardBuilderApproval,
	publicName string,
) error {
	if approval.PlanID == "" || approval.PlanDigest == "" || approval.PlanRevision < 1 ||
		approval.PlanETag == "" || !time.Now().UTC().Before(approval.ExpiresAt.UTC()) {
		return fmt.Errorf("Builder publication review omitted immutable confirmation metadata")
	}
	if approval.Summary.RecipeID == "" || approval.Summary.RecipeName == "" ||
		approval.Summary.EntrypointID == "" || approval.Summary.EntrypointName != publicName {
		return fmt.Errorf("Builder publication review does not describe the requested Recipe and Entrypoint")
	}
	return nil
}

func collectDashboardBuilderReviewEvidence(
	events []dashboardBuilderEvent,
	turnID string,
	targetModel string,
) (*dashboardBuilderReviewEvidence, error) {
	evidence := &dashboardBuilderReviewEvidence{
		completed:            make(map[string]int, len(dashboardBuilderRequiredTools)),
		targetModelIDs:       make(map[string]struct{}),
		createInvocations:    make(map[string]string),
		createdRecipeIDs:     make(map[string]struct{}),
		createdEntrypointIDs: make(map[string]struct{}),
	}
	for _, event := range events {
		if event.TurnID != turnID {
			continue
		}
		switch event.Type {
		case "tool_request":
			if err := evidence.recordToolRequest(event.Payload); err != nil {
				return nil, err
			}
		case "tool_result":
			if err := evidence.recordToolResult(event.Payload, targetModel); err != nil {
				return nil, err
			}
		}
	}
	return evidence, nil
}

func (evidence *dashboardBuilderReviewEvidence) recordToolRequest(payload json.RawMessage) error {
	var request dashboardBuilderToolRequest
	if err := json.Unmarshal(payload, &request); err != nil {
		return fmt.Errorf("decode Builder tool request: %w", err)
	}
	if request.ToolName != "router.recipe.prepare" && request.ToolName != "router.entrypoint.prepare" {
		return nil
	}
	var arguments struct {
		ExpectedRevision int64 `json:"expectedRevision"`
	}
	if err := json.Unmarshal(request.Arguments, &arguments); err != nil {
		return fmt.Errorf("decode Builder draft request: %w", err)
	}
	if arguments.ExpectedRevision == 0 {
		evidence.createInvocations[request.InvocationID] = request.ToolName
	}
	return nil
}

func (evidence *dashboardBuilderReviewEvidence) recordToolResult(
	payload json.RawMessage,
	targetModel string,
) error {
	var result dashboardBuilderToolResult
	if err := json.Unmarshal(payload, &result); err != nil {
		return fmt.Errorf("decode Builder tool result: %w", err)
	}
	if result.Status != "completed" || result.Error != nil {
		return fmt.Errorf(
			"Builder tool %q did not complete cleanly (status=%q error=%v)",
			result.ToolName, result.Status, result.Error,
		)
	}
	evidence.completed[result.ToolName]++
	switch result.ToolName {
	case "router.recipe.prepare":
		return evidence.recordPreparedRecipe(result)
	case "router.entrypoint.prepare":
		return evidence.recordPreparedEntrypoint(result)
	case "router.models.list":
		return evidence.recordModelDiscovery(result.Result, targetModel)
	case "router.recipe.validate":
		return evidence.recordValidation(result.Result)
	case "router.recipe.probe":
		return evidence.recordProbe(result.Result)
	case "router.recipe.evaluate":
		return evidence.recordEvaluation(result.Result)
	default:
		return nil
	}
}

func (evidence *dashboardBuilderReviewEvidence) recordPreparedRecipe(
	result dashboardBuilderToolResult,
) error {
	var value struct {
		RecipeID string `json:"recipeId"`
	}
	if err := json.Unmarshal(result.Result, &value); err != nil {
		return fmt.Errorf("decode prepared Recipe: %w", err)
	}
	if evidence.createInvocations[result.InvocationID] == result.ToolName && value.RecipeID != "" {
		evidence.createdRecipeIDs[value.RecipeID] = struct{}{}
	}
	return nil
}

func (evidence *dashboardBuilderReviewEvidence) recordPreparedEntrypoint(
	result dashboardBuilderToolResult,
) error {
	var value struct {
		EntrypointID string `json:"entrypointId"`
	}
	if err := json.Unmarshal(result.Result, &value); err != nil {
		return fmt.Errorf("decode prepared Entrypoint: %w", err)
	}
	if evidence.createInvocations[result.InvocationID] == result.ToolName && value.EntrypointID != "" {
		evidence.createdEntrypointIDs[value.EntrypointID] = struct{}{}
	}
	return nil
}

func (evidence *dashboardBuilderReviewEvidence) recordModelDiscovery(
	result json.RawMessage,
	targetModel string,
) error {
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
	if err := json.Unmarshal(result, &value); err != nil {
		return fmt.Errorf("decode connected Model discovery: %w", err)
	}
	for _, model := range value.Data {
		matchesTarget := model.ID == targetModel || model.Name == targetModel ||
			dashboardBuilderContains(model.Card.Aliases, targetModel)
		if model.Status == "active" && matchesTarget {
			evidence.targetModelIDs[model.ID] = struct{}{}
		}
	}
	return nil
}

func (evidence *dashboardBuilderReviewEvidence) recordValidation(result json.RawMessage) error {
	var value struct {
		Valid bool `json:"valid"`
	}
	if err := json.Unmarshal(result, &value); err != nil {
		return fmt.Errorf("decode Recipe validation evidence: %w", err)
	}
	evidence.validationPassed = evidence.validationPassed || value.Valid
	return nil
}

func (evidence *dashboardBuilderReviewEvidence) recordProbe(result json.RawMessage) error {
	passed, err := decodeDashboardBuilderPassed(result, "probe")
	if err != nil {
		return err
	}
	evidence.probePassed = evidence.probePassed || passed
	return nil
}

func (evidence *dashboardBuilderReviewEvidence) recordEvaluation(result json.RawMessage) error {
	passed, err := decodeDashboardBuilderPassed(result, "evaluation")
	if err != nil {
		return err
	}
	evidence.evaluationPassed = evidence.evaluationPassed || passed
	return nil
}

func decodeDashboardBuilderPassed(result json.RawMessage, evidenceName string) (bool, error) {
	var value struct {
		Passed bool `json:"passed"`
	}
	if err := json.Unmarshal(result, &value); err != nil {
		return false, fmt.Errorf("decode Recipe %s evidence: %w", evidenceName, err)
	}
	return value.Passed, nil
}

func (evidence *dashboardBuilderReviewEvidence) assertRequired(
	approval dashboardBuilderApproval,
) error {
	for _, toolName := range dashboardBuilderRequiredTools {
		if evidence.completed[toolName] == 0 {
			return fmt.Errorf("Builder publication review is missing completed tool %q", toolName)
		}
	}
	if !evidence.validationPassed || !evidence.probePassed || !evidence.evaluationPassed {
		return fmt.Errorf(
			"Builder evidence did not pass (validation=%t probe=%t evaluation=%t)",
			evidence.validationPassed, evidence.probePassed, evidence.evaluationPassed,
		)
	}
	if _, created := evidence.createdRecipeIDs[approval.Summary.RecipeID]; !created {
		return fmt.Errorf("Builder publication review did not use the Recipe created in this turn")
	}
	if _, created := evidence.createdEntrypointIDs[approval.Summary.EntrypointID]; !created {
		return fmt.Errorf("Builder publication review did not use the Entrypoint created in this turn")
	}
	return nil
}

func assertDashboardBuilderGates(raw json.RawMessage) error {
	var gates []struct {
		Name   string `json:"name"`
		Passed bool   `json:"passed"`
	}
	if err := json.Unmarshal(raw, &gates); err != nil {
		return fmt.Errorf("decode Builder publication gate results: %w", err)
	}
	if len(gates) == 0 {
		return fmt.Errorf("Builder publication review omitted gate results")
	}
	for _, gate := range gates {
		if gate.Name == "" || !gate.Passed {
			return fmt.Errorf("Builder publication gate did not pass: %+v", gate)
		}
	}
	return nil
}

func (evidence *dashboardBuilderReviewEvidence) assertAssignments(
	modelIDs []string,
	targetModel string,
) error {
	if len(modelIDs) == 0 {
		return fmt.Errorf("Builder publication review has no assigned Models")
	}
	if len(evidence.targetModelIDs) == 0 {
		return fmt.Errorf("Builder did not discover the real session target %q", targetModel)
	}
	for _, modelID := range modelIDs {
		if _, expected := evidence.targetModelIDs[modelID]; !expected {
			return fmt.Errorf(
				"Builder assigned Model %q instead of the real session target %q",
				modelID, targetModel,
			)
		}
	}
	return nil
}
