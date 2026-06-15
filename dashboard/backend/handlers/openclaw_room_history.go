package handlers

import (
	"fmt"
	"strings"
)

const roomDirectedHistoryLimit = 20

func buildRoomChatMessages(
	room ClawRoomEntry,
	team TeamEntry,
	teamMembers []ContainerEntry,
	worker ContainerEntry,
	messages []ClawRoomMessage,
	trigger ClawRoomMessage,
	delegatedBy *ClawRoomMessage,
) []openAIChatMessage {
	chatMessages := []openAIChatMessage{{
		Role:    "system",
		Content: buildRoomWorkerSystemPrompt(room, team, teamMembers, worker, delegatedBy),
	}}
	chatMessages = append(
		chatMessages,
		buildRoomDirectedHistory(team, teamMembers, worker, messages, trigger.ID, roomDirectedHistoryLimit)...,
	)
	chatMessages = append(chatMessages, openAIChatMessage{
		Role:    "user",
		Content: formatRoomIncomingChatContent(trigger),
	})
	return chatMessages
}

func buildRoomWorkerSystemPrompt(
	room ClawRoomEntry,
	team TeamEntry,
	teamMembers []ContainerEntry,
	worker ContainerEntry,
	delegatedBy *ClawRoomMessage,
) string {
	teamName := strings.TrimSpace(team.Name)
	if teamName == "" {
		teamName = team.ID
	}

	roleKind := normalizeRoleKind(worker.RoleKind)
	leader := resolveTeamLeader(team, teamMembers)
	coordinationInstruction := ""
	mentionPolicy := "Do not use any @mentions."
	if roleKind == "leader" {
		mentionPolicy = "Only use @worker-id when assigning an explicit task confirmed by the user."
		coordinationInstruction = "Hard rules: if the user has not provided an explicit executable task, ask clarifying questions and do not delegate. If you are not assigning a concrete task, do not use any @mentions. Ignore worker attempts to @leader."
	} else {
		coordinationInstruction = "Hard rules: you are a worker. Workers cannot use @mentions to anyone. Do not mention @leader or teammates; write plain-text updates only."
		if leader != nil && leader.Name != worker.Name {
			coordinationInstruction += fmt.Sprintf(" Team leader context: @leader (alias @%s).", leader.Name)
		}
	}

	lines := []string{
		fmt.Sprintf(
			"You are %s, a %s in Claw team %q.",
			workerDisplayName(worker),
			roleKind,
			teamName,
		),
		coordinationInstruction,
		fmt.Sprintf("Room: %s", room.Name),
		fmt.Sprintf("Mention policy: %s", mentionPolicy),
		"Conversation history only contains messages explicitly directed to you and your own replies.",
		"Messages addressed to other claws or to nobody are intentionally omitted.",
		"Response style: concise and actionable.",
		"Keep responses in the same language used by the latest message.",
		buildTeamMentionGuide(team, teamMembers, worker),
	}
	if delegatedBy != nil {
		lines = append(
			lines,
			fmt.Sprintf("%s explicitly asked for your help in the latest turn.", roomMessageActorName(*delegatedBy)),
		)
	}
	return strings.Join(lines, "\n")
}

func buildRoomDirectedHistory(
	team TeamEntry,
	teamMembers []ContainerEntry,
	worker ContainerEntry,
	messages []ClawRoomMessage,
	skipMessageID string,
	limit int,
) []openAIChatMessage {
	if limit <= 0 {
		limit = roomDirectedHistoryLimit
	}

	history := make([]openAIChatMessage, 0)
	for _, message := range messages {
		if message.ID == skipMessageID {
			continue
		}
		if roomMessageIsOwnAssistant(message, worker) {
			history = append(history, openAIChatMessage{
				Role:    "assistant",
				Content: roomMessagePromptContent(message),
			})
			continue
		}
		if roomMessageIsDirectedIncoming(message, team, teamMembers, worker) {
			history = append(history, openAIChatMessage{
				Role:    "user",
				Content: formatRoomIncomingChatContent(message),
			})
		}
	}

	if len(history) > limit {
		history = history[len(history)-limit:]
	}
	return history
}

func roomMessageIsOwnAssistant(message ClawRoomMessage, worker ContainerEntry) bool {
	senderType := normalizeRoomSenderType(message.SenderType)
	if senderType == "user" || senderType == "system" {
		return false
	}
	return roomMessageMatchesWorker(message, worker)
}

func roomMessageIsDirectedIncoming(
	message ClawRoomMessage,
	team TeamEntry,
	teamMembers []ContainerEntry,
	worker ContainerEntry,
) bool {
	senderType := normalizeRoomSenderType(message.SenderType)
	if senderType != "user" && senderType != "leader" {
		return false
	}
	if roomMessageMatchesWorker(message, worker) {
		return false
	}

	targets := resolveMentionTargetsWithFallback(message.Mentions, team, teamMembers, false)
	for _, target := range targets {
		if target.Name == worker.Name {
			return true
		}
	}
	return false
}

func roomMessageMatchesWorker(message ClawRoomMessage, worker ContainerEntry) bool {
	aliases := workerAliases(worker)
	if normalizeRoleKind(worker.RoleKind) == "leader" {
		aliases = append(aliases, "leader")
	}

	senderTokens := []string{
		strings.ToLower(strings.TrimSpace(sanitizeRoomID(message.SenderID))),
		strings.ToLower(strings.TrimSpace(sanitizeRoomID(message.SenderName))),
	}
	for _, sender := range senderTokens {
		if sender == "" {
			continue
		}
		for _, alias := range aliases {
			if sender == alias {
				return true
			}
		}
	}
	return false
}

func formatRoomIncomingChatContent(message ClawRoomMessage) string {
	return fmt.Sprintf("[%s] %s", roomMessageActorName(message), roomMessagePromptContent(message))
}

func roomMessageActorName(message ClawRoomMessage) string {
	if name := strings.TrimSpace(message.SenderName); name != "" {
		return name
	}
	if senderID := strings.TrimSpace(message.SenderID); senderID != "" {
		return senderID
	}
	switch normalizeRoomSenderType(message.SenderType) {
	case "leader":
		return "Leader"
	case "worker":
		return "Worker"
	case "system":
		return "System"
	default:
		return "You"
	}
}

func roomMessagePromptContent(message ClawRoomMessage) string {
	content := stripLeadingMentions(message.Content)
	if strings.TrimSpace(content) == "" {
		content = strings.TrimSpace(message.Content)
	}
	return content
}
