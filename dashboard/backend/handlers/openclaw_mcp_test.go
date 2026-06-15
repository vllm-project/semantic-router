package handlers

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
)

func decodeMCPToolJSONResult(t *testing.T, result *mcp.CallToolResult) any {
	t.Helper()

	if result == nil {
		t.Fatalf("tool result is nil")
	}
	if result.IsError {
		t.Fatalf("tool returned error result")
	}
	if len(result.Content) == 0 {
		t.Fatalf("tool result has no content")
	}

	text, ok := result.Content[0].(mcp.TextContent)
	if !ok {
		t.Fatalf("first content is not text: %#v", result.Content[0])
	}

	var payload any
	if err := json.Unmarshal([]byte(text.Text), &payload); err != nil {
		t.Fatalf("failed to decode tool result JSON: %v", err)
	}
	return payload
}

func decodeMCPToolTextResult(t *testing.T, result *mcp.CallToolResult) string {
	t.Helper()

	if result == nil {
		t.Fatalf("tool result is nil")
	}
	if len(result.Content) == 0 {
		t.Fatalf("tool result has no content")
	}

	text, ok := result.Content[0].(mcp.TextContent)
	if !ok {
		t.Fatalf("first content is not text: %#v", result.Content[0])
	}
	return text.Text
}

func TestOpenClawMCP_CreateAndListTeams(t *testing.T) {
	oc := newTestOpenClawHandler(t, t.TempDir(), false)
	handler := &OpenClawMCPHandler{openClaw: oc}

	createReq := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Arguments: map[string]any{
				"name":      "Alpha Team",
				"role":      "Routing",
				"principal": "Safety first",
			},
		},
	}

	createdResult, err := handler.createTeamTool(context.Background(), createReq)
	if err != nil {
		t.Fatalf("createTeamTool returned error: %v", err)
	}
	createdPayload := decodeMCPToolJSONResult(t, createdResult)
	createdMap, ok := createdPayload.(map[string]any)
	if !ok {
		t.Fatalf("unexpected create payload type: %T", createdPayload)
	}
	if createdMap["name"] != "Alpha Team" {
		t.Fatalf("expected team name Alpha Team, got %#v", createdMap["name"])
	}

	listResult, err := handler.listTeamsTool(context.Background(), mcp.CallToolRequest{
		Params: mcp.CallToolParams{Arguments: map[string]any{}},
	})
	if err != nil {
		t.Fatalf("listTeamsTool returned error: %v", err)
	}
	listPayload := decodeMCPToolJSONResult(t, listResult)
	teams, ok := listPayload.([]any)
	if !ok {
		t.Fatalf("unexpected list payload type: %T", listPayload)
	}
	if len(teams) != 1 {
		t.Fatalf("expected 1 team, got %d", len(teams))
	}
}

func TestOpenClawMCP_UpdateWorkerMergesIdentityFields(t *testing.T) {
	oc := newTestOpenClawHandler(t, t.TempDir(), false)
	handler := &OpenClawMCPHandler{openClaw: oc}

	team := TeamEntry{
		ID:        "team-alpha",
		Name:      "Team Alpha",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	if err := oc.saveTeams([]TeamEntry{team}); err != nil {
		t.Fatalf("failed to seed teams: %v", err)
	}

	worker := ContainerEntry{
		Name:            "worker-a",
		Port:            18788,
		Image:           "ghcr.io/openclaw/openclaw:latest",
		Token:           "token",
		DataDir:         "/tmp/worker-a",
		CreatedAt:       time.Now().UTC().Format(time.RFC3339),
		TeamID:          team.ID,
		TeamName:        team.Name,
		AgentName:       "Worker A",
		AgentEmoji:      "🦀",
		AgentRole:       "Analyst",
		AgentVibe:       "Calm",
		AgentPrinciples: "Be precise",
	}
	if err := oc.saveRegistry([]ContainerEntry{worker}); err != nil {
		t.Fatalf("failed to seed workers: %v", err)
	}

	updateReq := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Arguments: map[string]any{
				"worker_id": "worker-a",
				"role":      "Operator",
			},
		},
	}

	updatedResult, err := handler.updateWorkerTool(context.Background(), updateReq)
	if err != nil {
		t.Fatalf("updateWorkerTool returned error: %v", err)
	}
	updatedPayload := decodeMCPToolJSONResult(t, updatedResult)
	updatedMap, ok := updatedPayload.(map[string]any)
	if !ok {
		t.Fatalf("unexpected update payload type: %T", updatedPayload)
	}

	if updatedMap["agentRole"] != "Operator" {
		t.Fatalf("expected agentRole to be updated to Operator, got %#v", updatedMap["agentRole"])
	}
	if updatedMap["agentName"] != "Worker A" {
		t.Fatalf("expected agentName to be preserved, got %#v", updatedMap["agentName"])
	}
}

func TestOpenClawMCP_CreateWorkerRequiresIdentityFields(t *testing.T) {
	handler := &OpenClawMCPHandler{openClaw: newTestOpenClawHandler(t, t.TempDir(), false)}

	testCases := []struct {
		name      string
		arguments map[string]any
		wantError string
	}{
		{
			name: "emoji required",
			arguments: map[string]any{
				"team_id": "team-alpha",
				"name":    "worker-a",
			},
			wantError: "emoji is required",
		},
		{
			name: "role required",
			arguments: map[string]any{
				"team_id": "team-alpha",
				"name":    "worker-a",
				"emoji":   "🦞",
			},
			wantError: "role is required",
		},
		{
			name: "vibe required",
			arguments: map[string]any{
				"team_id": "team-alpha",
				"name":    "worker-a",
				"emoji":   "🦞",
				"role":    "operator",
			},
			wantError: "vibe is required",
		},
		{
			name: "principles required",
			arguments: map[string]any{
				"team_id": "team-alpha",
				"name":    "worker-a",
				"emoji":   "🦞",
				"role":    "operator",
				"vibe":    "calm",
			},
			wantError: "principles is required",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := handler.createWorkerTool(context.Background(), mcp.CallToolRequest{
				Params: mcp.CallToolParams{Arguments: tc.arguments},
			})
			if err != nil {
				t.Fatalf("createWorkerTool returned error: %v", err)
			}
			if result == nil || !result.IsError {
				t.Fatalf("expected error result, got %#v", result)
			}

			message := decodeMCPToolTextResult(t, result)
			if !strings.Contains(message, tc.wantError) {
				t.Fatalf("expected error message to contain %q, got %q", tc.wantError, message)
			}
		})
	}
}

func TestOpenClawMCP_CreateWorkerUsesServerDefaultsForRuntimeFields(t *testing.T) {
	oc := newTestOpenClawHandler(t, t.TempDir(), true)
	handler := &OpenClawMCPHandler{openClaw: oc}

	result, err := handler.createWorkerTool(context.Background(), mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Arguments: map[string]any{
				"team_id":    "team-alpha",
				"name":       "worker-a",
				"emoji":      "🦞",
				"role":       "operator",
				"vibe":       "calm",
				"principles": "be precise",
			},
		},
	})
	if err != nil {
		t.Fatalf("createWorkerTool returned error: %v", err)
	}
	if result == nil || !result.IsError {
		t.Fatalf("expected read-only error result, got %#v", result)
	}

	message := decodeMCPToolTextResult(t, result)
	if !strings.Contains(message, "Read-only mode enabled") {
		t.Fatalf("expected read-only error, got %q", message)
	}
}

func TestClawCreateWorkerToolDefinition_HireClawGuidance(t *testing.T) {
	tool := clawCreateWorkerToolDefinition()

	if tool.Name != "claw_create_worker" {
		t.Fatalf("unexpected tool name: %q", tool.Name)
	}
	if !strings.Contains(tool.Description, "Hire/create") {
		t.Fatalf("expected HireClaw-oriented description, got %q", tool.Description)
	}

	required := tool.InputSchema.Required
	for _, field := range []string{"team_id", "name", "emoji", "role", "vibe", "principles"} {
		if !slices.Contains(required, field) {
			t.Fatalf("expected required field %q in %v", field, required)
		}
	}

	assertDescriptionContains := func(field string, expected ...string) {
		t.Helper()

		property, ok := tool.InputSchema.Properties[field].(map[string]any)
		if !ok {
			t.Fatalf("expected schema property for %q, got %#v", field, tool.InputSchema.Properties[field])
		}
		description, ok := property["description"].(string)
		if !ok {
			t.Fatalf("expected string description for %q, got %#v", field, property["description"])
		}
		for _, want := range expected {
			if !strings.Contains(description, want) {
				t.Fatalf("expected description for %q to contain %q, got %q", field, want, description)
			}
		}
	}

	assertDescriptionContains("name", "Human-facing worker identity name", "English first name")
	assertDescriptionContains("role", "Concrete functional specialty")
	assertDescriptionContains("vibe", "human collaboration style", "pressure behavior")
	assertDescriptionContains("principles", "Team-aware operating rules", "leader coordination", "teammate expectations")
}

func TestOpenClawMCP_UpdateWorkerRoleKindPromotesLeader(t *testing.T) {
	oc := newTestOpenClawHandler(t, t.TempDir(), false)
	handler := &OpenClawMCPHandler{openClaw: oc}

	team := TeamEntry{
		ID:        "team-leader",
		Name:      "Team Leader",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	if err := oc.saveTeams([]TeamEntry{team}); err != nil {
		t.Fatalf("failed to seed teams: %v", err)
	}
	if err := oc.saveRegistry([]ContainerEntry{
		{
			Name:      "worker-a",
			Port:      18788,
			Image:     "ghcr.io/openclaw/openclaw:latest",
			Token:     "token-a",
			DataDir:   "/tmp/worker-a",
			CreatedAt: time.Now().UTC().Format(time.RFC3339),
			TeamID:    team.ID,
			TeamName:  team.Name,
			RoleKind:  "worker",
		},
		{
			Name:      "worker-b",
			Port:      18789,
			Image:     "ghcr.io/openclaw/openclaw:latest",
			Token:     "token-b",
			DataDir:   "/tmp/worker-b",
			CreatedAt: time.Now().UTC().Format(time.RFC3339),
			TeamID:    team.ID,
			TeamName:  team.Name,
			RoleKind:  "leader",
		},
	}); err != nil {
		t.Fatalf("failed to seed workers: %v", err)
	}

	leaderKind := "leader"
	updateResult, err := handler.updateWorkerTool(context.Background(), mcp.CallToolRequest{
		Params: mcp.CallToolParams{Arguments: map[string]any{
			"worker_id": "worker-a",
			"role_kind": leaderKind,
		}},
	})
	if err != nil {
		t.Fatalf("updateWorkerTool returned error: %v", err)
	}
	updatePayload := decodeMCPToolJSONResult(t, updateResult)
	updateMap, ok := updatePayload.(map[string]any)
	if !ok {
		t.Fatalf("unexpected update payload type: %T", updatePayload)
	}
	if updateMap["roleKind"] != "leader" {
		t.Fatalf("expected roleKind=leader, got %#v", updateMap["roleKind"])
	}

	teams, err := oc.loadTeams()
	if err != nil {
		t.Fatalf("failed to load teams: %v", err)
	}
	if len(teams) != 1 || teams[0].LeaderID != "worker-a" {
		t.Fatalf("expected team leader to be worker-a, got %+v", teams)
	}
}
