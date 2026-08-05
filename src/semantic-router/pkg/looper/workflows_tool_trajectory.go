package looper

func cloneWorkflowToolTrajectories(
	src map[string][]workflowAgentToolTurn,
) map[string][]workflowAgentToolTurn {
	if len(src) == 0 {
		return nil
	}
	cloned := make(map[string][]workflowAgentToolTurn, len(src))
	for agentID, turns := range src {
		cloned[agentID] = cloneWorkflowToolTrajectory(turns)
	}
	return cloned
}

func cloneWorkflowToolTrajectory(src []workflowAgentToolTurn) []workflowAgentToolTurn {
	if len(src) == 0 {
		return nil
	}
	cloned := make([]workflowAgentToolTurn, 0, len(src))
	for _, turn := range src {
		cloned = append(cloned, workflowAgentToolTurn{
			AgentID:      turn.AgentID,
			Phase:        turn.Phase,
			StepID:       turn.StepID,
			Role:         turn.Role,
			Model:        turn.Model,
			ToolCallIDs:  append([]string(nil), turn.ToolCallIDs...),
			AssistantRaw: append([]byte(nil), turn.AssistantRaw...),
			ToolMessages: cloneWorkflowMessages(turn.ToolMessages),
		})
	}
	return cloned
}

func workflowStepToolTrajectoriesWithAgent(
	src map[string][]workflowAgentToolTurn,
	agentID string,
	turns []workflowAgentToolTurn,
) map[string][]workflowAgentToolTurn {
	merged := cloneWorkflowToolTrajectories(src)
	if len(turns) == 0 || agentID == "" {
		return merged
	}
	if merged == nil {
		merged = map[string][]workflowAgentToolTurn{}
	}
	merged[agentID] = cloneWorkflowToolTrajectory(turns)
	return merged
}

func workflowPendingStepToolTrajectories(state *workflowPendingToolState) map[string][]workflowAgentToolTurn {
	if state == nil {
		return nil
	}
	return workflowStepToolTrajectoriesWithAgent(
		state.CurrentStepToolTrajectories,
		state.AgentID,
		state.ToolTrajectory,
	)
}

func workflowToolTurnTraces(turns []workflowAgentToolTurn) []workflowToolTurnTrace {
	if len(turns) == 0 {
		return nil
	}
	traces := make([]workflowToolTurnTrace, 0, len(turns))
	for _, turn := range turns {
		traces = append(traces, workflowToolTurnTrace{
			ToolCallIDs: append([]string(nil), turn.ToolCallIDs...),
		})
	}
	return traces
}
