package handlers

import "github.com/mark3labs/mcp-go/mcp"

func clawListTeamsToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        "claw_list_teams",
		Description: "List all Claw teams.",
		InputSchema: mcp.ToolInputSchema{
			Type:       "object",
			Properties: map[string]any{},
		},
	}
}

func clawCreateTeamToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        "claw_create_team",
		Description: "Create a new Claw team.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"id": map[string]any{
					"type":        "string",
					"description": "Optional team ID. If omitted, one is derived from name.",
				},
				"name": map[string]any{
					"type":        "string",
					"description": "Team name.",
				},
				"vibe": map[string]any{
					"type":        "string",
					"description": "Team vibe/working style.",
				},
				"role": map[string]any{
					"type":        "string",
					"description": "Team role.",
				},
				"principal": map[string]any{
					"type":        "string",
					"description": "Team principal.",
				},
				"description": map[string]any{
					"type":        "string",
					"description": "Team description.",
				},
				"leader_id": map[string]any{
					"type":        "string",
					"description": "Optional worker id to set as team leader.",
				},
			},
			Required: []string{"name"},
		},
	}
}

func clawUpdateTeamToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        "claw_update_team",
		Description: "Update an existing Claw team.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"team_id": map[string]any{
					"type":        "string",
					"description": "Team ID to update.",
				},
				"name": map[string]any{
					"type":        "string",
					"description": "New team name.",
				},
				"vibe": map[string]any{
					"type":        "string",
					"description": "New team vibe.",
				},
				"role": map[string]any{
					"type":        "string",
					"description": "New team role.",
				},
				"principal": map[string]any{
					"type":        "string",
					"description": "New team principal.",
				},
				"description": map[string]any{
					"type":        "string",
					"description": "New team description.",
				},
				"leader_id": map[string]any{
					"type":        "string",
					"description": "Optional worker id to set as team leader. Use empty string to clear leader.",
				},
			},
			Required: []string{"team_id"},
		},
	}
}

func clawDeleteTeamToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        "claw_delete_team",
		Description: "Delete a Claw team by team_id.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"team_id": map[string]any{
					"type":        "string",
					"description": "Team ID to delete.",
				},
			},
			Required: []string{"team_id"},
		},
	}
}

func clawListWorkersToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        "claw_list_workers",
		Description: "List all Claw workers.",
		InputSchema: mcp.ToolInputSchema{
			Type:       "object",
			Properties: map[string]any{},
		},
	}
}

func clawGetWorkerToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        "claw_get_worker",
		Description: "Get one Claw worker by worker_id.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"worker_id": map[string]any{
					"type":        "string",
					"description": "Worker ID.",
				},
			},
			Required: []string{"worker_id"},
		},
	}
}

func clawCreateWorkerToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        "claw_create_worker",
		Description: "Hire/create a Claw worker and assign it to a team.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"team_id": map[string]any{
					"type":        "string",
					"description": "Target team ID.",
				},
				"name": map[string]any{
					"type":        "string",
					"description": "Human-facing worker identity name. Prefer a recruiter-grade English first name unless the user already specifies another name.",
				},
				"emoji": map[string]any{
					"type":        "string",
					"description": "Worker emoji.",
				},
				"role": map[string]any{
					"type":        "string",
					"description": "Concrete functional specialty for the worker.",
				},
				"vibe": map[string]any{
					"type":        "string",
					"description": "Specific human collaboration style, including temperament, communication rhythm, and pressure behavior.",
				},
				"principles": map[string]any{
					"type":        "string",
					"description": "Team-aware operating rules that cover mission fit, leader coordination, teammate expectations, and escalation boundaries.",
				},
				"role_kind": map[string]any{
					"type":        "string",
					"description": "Optional role kind. Allowed values: leader, worker.",
				},
				"skills": map[string]any{
					"type":        "array",
					"description": "Optional skill IDs to inject into worker workspace.",
					"items": map[string]any{
						"type": "string",
					},
				},
			},
			Required: []string{"team_id", "name", "emoji", "role", "vibe", "principles"},
		},
	}
}

func clawUpdateWorkerToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        "claw_update_worker",
		Description: "Update Claw worker team assignment and/or identity fields.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"worker_id": map[string]any{
					"type":        "string",
					"description": "Worker ID.",
				},
				"team_id": map[string]any{
					"type":        "string",
					"description": "Optional new team ID.",
				},
				"name": map[string]any{
					"type":        "string",
					"description": "Optional new identity name.",
				},
				"emoji": map[string]any{
					"type":        "string",
					"description": "Optional new identity emoji.",
				},
				"role": map[string]any{
					"type":        "string",
					"description": "Optional new identity role.",
				},
				"vibe": map[string]any{
					"type":        "string",
					"description": "Optional new identity vibe.",
				},
				"principles": map[string]any{
					"type":        "string",
					"description": "Optional new identity principles.",
				},
				"role_kind": map[string]any{
					"type":        "string",
					"description": "Optional role kind update. Allowed values: leader, worker.",
				},
			},
			Required: []string{"worker_id"},
		},
	}
}

func clawDeleteWorkerToolDefinition() mcp.Tool {
	return mcp.Tool{
		Name:        "claw_delete_worker",
		Description: "Delete a Claw worker by worker_id.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"worker_id": map[string]any{
					"type":        "string",
					"description": "Worker ID.",
				},
			},
			Required: []string{"worker_id"},
		},
	}
}
