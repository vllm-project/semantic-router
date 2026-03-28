package handlers

import (
	"errors"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"
)

const roomAutomationMaxTurns = 24

type roomAutomationContext struct {
	room     *ClawRoomEntry
	team     *TeamEntry
	entries  []ContainerEntry
	messages []ClawRoomMessage
}

type roomAutomationDispatch struct {
	trigger          ClawRoomMessage
	senderType       string
	workers          []ContainerEntry
	targets          []ContainerEntry
	snapshotMessages []ClawRoomMessage
}

type targetReplyResult struct {
	target ContainerEntry
	reply  ClawRoomMessage
	err    error
}

func (h *OpenClawHandler) roomAutomationLock(roomID string) *sync.Mutex {
	if existing, ok := h.roomAutomationMu.Load(roomID); ok {
		return existing.(*sync.Mutex)
	}
	lock := &sync.Mutex{}
	actual, _ := h.roomAutomationMu.LoadOrStore(roomID, lock)
	return actual.(*sync.Mutex)
}

func roomMessageAutomationProcessed(message ClawRoomMessage) bool {
	if message.Metadata == nil {
		return false
	}
	return strings.TrimSpace(message.Metadata[roomAutomationProcessedAtKey]) != ""
}

func (h *OpenClawHandler) markRoomMessageAutomationProcessed(roomID, messageID string) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	messages, err := h.loadRoomMessages(roomID)
	if err != nil {
		return err
	}

	changed := false
	for i := range messages {
		if messages[i].ID != messageID {
			continue
		}
		if roomMessageAutomationProcessed(messages[i]) {
			return nil
		}
		if messages[i].Metadata == nil {
			messages[i].Metadata = map[string]string{}
		}
		messages[i].Metadata[roomAutomationProcessedAtKey] = time.Now().UTC().Format(time.RFC3339Nano)
		changed = true
		break
	}

	if !changed {
		return nil
	}
	return h.saveRoomMessages(roomID, messages)
}

func (h *OpenClawHandler) processRoomUserMessage(roomID string, triggerMessageID string) {
	lock := h.roomAutomationLock(roomID)
	lock.Lock()
	defer lock.Unlock()

	queue := []string{strings.TrimSpace(triggerMessageID)}
	seen := map[string]bool{}
	turns := 0

	for len(queue) > 0 && turns < roomAutomationMaxTurns {
		currentID := strings.TrimSpace(queue[0])
		queue = queue[1:]
		if currentID == "" || seen[currentID] {
			continue
		}

		seen[currentID] = true
		turns++

		nextIDs, fatal := h.processRoomAutomationTurn(roomID, currentID)
		if fatal {
			return
		}
		queue = append(queue, nextIDs...)
	}

	h.appendRoomAutomationMaxTurnWarning(roomID, turns, queue)
}

func (h *OpenClawHandler) processRoomAutomationTurn(roomID, currentID string) ([]string, bool) {
	ctx, err := h.loadRoomAutomationContext(roomID)
	if err != nil {
		log.Printf("openclaw: room automation prefetch failed room=%s err=%v", roomID, err)
		return nil, true
	}
	if ctx.room == nil || ctx.team == nil {
		return nil, true
	}

	dispatch, ok := h.prepareRoomAutomationDispatch(roomID, currentID, ctx)
	if !ok {
		return nil, false
	}

	return h.dispatchRoomAutomationReplies(roomID, ctx, dispatch), false
}

func (h *OpenClawHandler) loadRoomAutomationContext(roomID string) (roomAutomationContext, error) {
	h.mu.RLock()
	rooms, roomErr := h.loadRooms()
	teams, teamErr := h.loadTeams()
	entries, entryErr := h.loadRegistry()
	messages, msgErr := h.loadRoomMessages(roomID)
	h.mu.RUnlock()
	if roomErr != nil || teamErr != nil || entryErr != nil || msgErr != nil {
		return roomAutomationContext{}, fmt.Errorf("room automation prefetch failed: %w", errors.Join(roomErr, teamErr, entryErr, msgErr))
	}

	room := findRoomByID(rooms, roomID)
	if room == nil {
		return roomAutomationContext{}, nil
	}

	return roomAutomationContext{
		room:     room,
		team:     findTeamByID(teams, room.TeamID),
		entries:  entries,
		messages: messages,
	}, nil
}

func findRoomMessageByID(messages []ClawRoomMessage, messageID string) *ClawRoomMessage {
	for i := range messages {
		if messages[i].ID == messageID {
			return &messages[i]
		}
	}
	return nil
}

func (h *OpenClawHandler) prepareRoomAutomationDispatch(
	roomID, currentID string,
	ctx roomAutomationContext,
) (*roomAutomationDispatch, bool) {
	trigger := findRoomMessageByID(ctx.messages, currentID)
	if trigger == nil {
		return nil, false
	}

	senderType := normalizeRoomSenderType(trigger.SenderType)
	if senderType == "system" || roomMessageAutomationProcessed(*trigger) {
		return nil, false
	}

	if err := h.markRoomMessageAutomationProcessed(roomID, trigger.ID); err != nil {
		log.Printf("openclaw: failed to mark room message as processed room=%s message=%s err=%v", roomID, trigger.ID, err)
	}
	if senderType == "worker" {
		return nil, false
	}

	workers := teamWorkers(ctx.entries, ctx.team.ID)
	targets := roomAutomationTargets(*ctx.team, workers, *trigger, senderType)
	if len(targets) == 0 {
		return nil, false
	}

	return &roomAutomationDispatch{
		trigger:          *trigger,
		senderType:       senderType,
		workers:          workers,
		targets:          targets,
		snapshotMessages: append([]ClawRoomMessage(nil), ctx.messages...),
	}, true
}

func roomAutomationTargets(
	team TeamEntry,
	workers []ContainerEntry,
	trigger ClawRoomMessage,
	senderType string,
) []ContainerEntry {
	targets := resolveMentionTargetsWithFallback(trigger.Mentions, team, workers, false)
	if senderType == "user" {
		return targets
	}

	leader := resolveTeamLeader(team, workers)
	filtered := make([]ContainerEntry, 0, len(targets))
	for _, target := range targets {
		isLeaderTarget := normalizeRoleKind(target.RoleKind) == "leader"
		if leader != nil && target.Name == leader.Name {
			isLeaderTarget = true
		}
		if isLeaderTarget {
			continue
		}
		filtered = append(filtered, target)
	}
	return filtered
}

func (h *OpenClawHandler) dispatchRoomAutomationReplies(
	roomID string,
	ctx roomAutomationContext,
	dispatch *roomAutomationDispatch,
) []string {
	results := make(chan targetReplyResult, len(dispatch.targets))
	expected := 0
	triggerSenderID := sanitizeContainerName(dispatch.trigger.SenderID)

	for _, target := range dispatch.targets {
		if triggerSenderID != "" && target.Name == triggerSenderID {
			continue
		}
		expected++
		h.startRoomAutomationReply(roomID, *ctx.room, *ctx.team, dispatch, target, results)
	}

	return h.collectRoomAutomationReplies(ctx.room.ID, *ctx.room, results, expected)
}

func (h *OpenClawHandler) startRoomAutomationReply(
	roomID string,
	room ClawRoomEntry,
	team TeamEntry,
	dispatch *roomAutomationDispatch,
	target ContainerEntry,
	results chan<- targetReplyResult,
) {
	var delegatedBy *ClawRoomMessage
	if dispatch.senderType == "leader" {
		triggerCopy := dispatch.trigger
		delegatedBy = &triggerCopy
	}

	placeholderID := generateRoomEntityID("room-msg")
	go func() {
		onChunk := func(chunk string, done bool) {
			h.publishRoomWSEvent(roomID, WSOutboundMessage{
				Type:      "message_chunk",
				MessageID: placeholderID,
				Status:    "streaming",
			})
		}

		reply, err := h.runWorkerReplyStream(
			room,
			team,
			dispatch.workers,
			target,
			dispatch.snapshotMessages,
			dispatch.trigger,
			delegatedBy,
			onChunk,
		)
		if err == nil {
			reply.ID = placeholderID
		}
		results <- targetReplyResult{
			target: target,
			reply:  reply,
			err:    err,
		}
	}()
}

func (h *OpenClawHandler) collectRoomAutomationReplies(
	roomID string,
	room ClawRoomEntry,
	results <-chan targetReplyResult,
	expected int,
) []string {
	next := make([]string, 0)
	for i := 0; i < expected; i++ {
		result := <-results
		if result.err != nil {
			h.appendRoomAutomationError(room, result.target, result.err)
			continue
		}

		if err := h.appendRoomMessage(roomID, result.reply); err != nil {
			log.Printf("openclaw: failed to append room reply: %v", err)
			continue
		}
		if len(result.reply.Mentions) > 0 {
			next = append(next, result.reply.ID)
		}
	}
	return next
}

func (h *OpenClawHandler) appendRoomAutomationError(room ClawRoomEntry, target ContainerEntry, replyErr error) {
	errMsg := newRoomMessage(
		room,
		"system",
		"clawos-system",
		"ClawOS",
		fmt.Sprintf("@%s is unavailable: %v", target.Name, replyErr),
		map[string]string{"worker": target.Name, "phase": "reply"},
	)
	if appendErr := h.appendRoomMessage(room.ID, errMsg); appendErr != nil {
		log.Printf("openclaw: failed to append room system message: %v", appendErr)
	}
}

func (h *OpenClawHandler) appendRoomAutomationMaxTurnWarning(roomID string, turns int, queue []string) {
	if turns < roomAutomationMaxTurns || len(queue) == 0 {
		return
	}

	h.mu.RLock()
	rooms, _ := h.loadRooms()
	h.mu.RUnlock()

	room := findRoomByID(rooms, roomID)
	if room == nil {
		return
	}

	warn := newRoomMessage(
		*room,
		"system",
		"clawos-system",
		"ClawOS",
		"Room automation reached max turns and was paused to avoid loops.",
		map[string]string{"phase": "safety", "reason": "max-turns"},
	)
	_ = h.appendRoomMessage(roomID, warn)
}
