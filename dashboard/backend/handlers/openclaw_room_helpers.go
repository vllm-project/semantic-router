package handlers

import (
	"fmt"
	"sort"
	"strings"
	"unicode"
)

type openAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func teamWorkers(entries []ContainerEntry, teamID string) []ContainerEntry {
	workers := make([]ContainerEntry, 0)
	for _, entry := range entries {
		if entry.TeamID != teamID {
			continue
		}
		entry.RoleKind = normalizeRoleKind(entry.RoleKind)
		workers = append(workers, enrichContainerIdentity(entry))
	}
	sort.Slice(workers, func(i, j int) bool { return workers[i].Name < workers[j].Name })
	return workers
}

func resolveTeamLeader(team TeamEntry, workers []ContainerEntry) *ContainerEntry {
	leaderID := strings.TrimSpace(team.LeaderID)
	if leaderID != "" {
		for i := range workers {
			if workers[i].Name == leaderID {
				return &workers[i]
			}
		}
	}
	for i := range workers {
		if normalizeRoleKind(workers[i].RoleKind) == "leader" {
			return &workers[i]
		}
	}
	return nil
}

func workerAliases(worker ContainerEntry) []string {
	aliases := []string{strings.ToLower(strings.TrimSpace(worker.Name))}
	if alias := strings.ToLower(strings.TrimSpace(sanitizeRoomID(worker.AgentName))); alias != "" {
		aliases = append(aliases, alias)
	}
	return aliases
}

func resolveMentionTargetsWithFallback(
	mentions []string,
	team TeamEntry,
	workers []ContainerEntry,
	defaultToLeader bool,
) []ContainerEntry {
	if len(workers) == 0 {
		return nil
	}

	leader := resolveTeamLeader(team, workers)
	lookup := map[string]ContainerEntry{}
	for _, worker := range workers {
		for _, alias := range workerAliases(worker) {
			lookup[alias] = worker
		}
	}

	picked := map[string]ContainerEntry{}
	for _, mention := range mentions {
		token := strings.ToLower(strings.TrimSpace(mention))
		if token == "" {
			continue
		}
		if token == "all" {
			for _, worker := range workers {
				picked[worker.Name] = worker
			}
			continue
		}
		if token == "leader" {
			if leader != nil {
				picked[leader.Name] = *leader
			}
			continue
		}
		if worker, ok := lookup[token]; ok {
			picked[worker.Name] = worker
		}
	}

	if len(picked) == 0 && defaultToLeader && leader != nil {
		picked[leader.Name] = *leader
	}

	out := make([]ContainerEntry, 0, len(picked))
	for _, worker := range picked {
		out = append(out, worker)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func stripLeadingMentions(content string) string {
	rawRunes := []rune(strings.TrimSpace(content))
	if len(rawRunes) == 0 {
		return ""
	}

	// Remove addressing prefixes like:
	// "@a @b task", "@leader，安排一下", "@x,@y do this"
	i := 0
	consumedMention := false
	for i < len(rawRunes) {
		if rawRunes[i] != '@' {
			break
		}

		j := i + 1
		for j < len(rawRunes) {
			ch := rawRunes[j]
			if unicode.IsLetter(ch) || unicode.IsDigit(ch) || ch == '_' || ch == '.' || ch == '-' {
				j++
				continue
			}
			break
		}

		if j == i+1 {
			return string(rawRunes)
		}

		consumedMention = true
		i = j
		for i < len(rawRunes) {
			ch := rawRunes[i]
			if unicode.IsSpace(ch) || ch == ',' || ch == ';' || ch == '，' || ch == '；' || ch == '、' {
				i++
				continue
			}
			break
		}
	}

	if !consumedMention {
		return string(rawRunes)
	}

	trimmed := strings.TrimSpace(string(rawRunes[i:]))
	if trimmed == "" {
		return string(rawRunes)
	}
	return trimmed
}

func buildTeamMentionGuide(team TeamEntry, workers []ContainerEntry, self ContainerEntry) string {
	if len(workers) == 0 {
		return "No teammates registered."
	}

	sorted := append([]ContainerEntry(nil), workers...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })

	leader := resolveTeamLeader(team, workers)
	lines := make([]string, 0, len(sorted)+5)
	if leader != nil {
		leaderRole := strings.TrimSpace(leader.AgentRole)
		if leaderRole == "" {
			leaderRole = "leader"
		}
		lines = append(
			lines,
			fmt.Sprintf(
				"Leader aliases: @leader and @%s = %s (%s)",
				leader.Name,
				workerDisplayName(*leader),
				leaderRole,
			),
		)
		if leader.Name == self.Name {
			lines = append(lines, "You are the leader. Delegate with @worker-id mentions and keep the team aligned.")
			lines = append(lines, "Hard rule: do not delegate with @mentions until the user gives an explicit executable task.")
		} else {
			lines = append(
				lines,
				fmt.Sprintf("Your leader is @leader (same as @%s).", leader.Name),
			)
			lines = append(lines, "Hard rule: workers cannot use @mentions. Report progress in plain text without @leader or @worker-id.")
		}
	} else {
		lines = append(lines, "No leader is assigned yet. Coordinate directly with @worker-id aliases.")
	}

	lines = append(lines, "Team member aliases for delegation:")
	for _, member := range sorted {
		roleKind := normalizeRoleKind(member.RoleKind)
		if roleKind != "leader" {
			roleKind = "worker"
		}
		roleText := strings.TrimSpace(member.AgentRole)
		if roleText == "" {
			roleText = roleKind
		}
		displayName := workerDisplayName(member)
		line := fmt.Sprintf("- @%s = %s (%s)", member.Name, displayName, roleText)
		if member.Name == self.Name {
			line += " [you]"
		}
		lines = append(lines, line)
	}
	lines = append(lines, "Only leader can delegate with @worker-id, and only after explicit user task confirmation.")
	return strings.Join(lines, "\n")
}
