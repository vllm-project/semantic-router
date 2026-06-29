package routerreplay

import "sort"

// ListSummaryRecord returns a copy of rec with large captured fields cleared so
// replay list responses are restricted in size.
// Full payloads remain available via GET /v1/router_replay/{id}.
func ListSummaryRecord(rec RoutingRecord) RoutingRecord {
	out := rec
	out.RequestBody = ""
	out.ResponseBody = ""
	out.Prompt = ""
	out.ToolDefinitions = ""
	out.ProjectionTrace = nil
	out.ToolTrace = toolTraceNamesOnly(out.ToolTrace)
	return out
}

func toolTraceNamesOnly(tt *ToolTrace) *ToolTrace {
	if tt == nil {
		return nil
	}
	seen := make(map[string]struct{})
	for _, n := range tt.ToolNames {
		if n == "" {
			continue
		}
		seen[n] = struct{}{}
	}
	for _, step := range tt.Steps {
		if step.ToolName == "" {
			continue
		}
		seen[step.ToolName] = struct{}{}
	}
	names := make([]string, 0, len(seen))
	for n := range seen {
		names = append(names, n)
	}
	sort.Strings(names)
	return &ToolTrace{
		Flow:      tt.Flow,
		Stage:     tt.Stage,
		ToolNames: names,
	}
}
