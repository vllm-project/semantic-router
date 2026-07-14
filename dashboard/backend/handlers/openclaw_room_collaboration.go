package handlers

import (
	"log"
	"sync"
	"time"
)

// ClawRoomCollaborationEvent is the canonical room-scoped live event envelope.
type ClawRoomCollaborationEvent struct {
	Type            string           `json:"type"`
	RoomID          string           `json:"roomId,omitempty"`
	Message         *ClawRoomMessage `json:"message,omitempty"`
	MessageID       string           `json:"messageId,omitempty"`
	Chunk           string           `json:"chunk,omitempty"`
	Status          string           `json:"status,omitempty"`
	Error           string           `json:"error,omitempty"`
	ParticipantType string           `json:"participantType,omitempty"`
	ParticipantID   string           `json:"participantId,omitempty"`
	SessionUser     string           `json:"sessionUser,omitempty"`
	Payload         map[string]any   `json:"payload,omitempty"`
	Timestamp       string           `json:"timestamp,omitempty"`
}

func cloneCollaborationEvent(event ClawRoomCollaborationEvent) ClawRoomCollaborationEvent {
	if event.Message == nil {
		return event
	}
	copyMsg := *event.Message
	event.Message = &copyMsg
	return event
}

func newMessageCollaborationEvent(message ClawRoomMessage) ClawRoomCollaborationEvent {
	copyMsg := message
	participantType := normalizeRoomSenderType(message.SenderType)
	return ClawRoomCollaborationEvent{
		Type:            WSTypeNewMessage,
		Message:         &copyMsg,
		ParticipantType: participantType,
		ParticipantID:   message.SenderID,
	}
}

func messageUpdatedCollaborationEvent(room ClawRoomEntry, message ClawRoomMessage) ClawRoomCollaborationEvent {
	copyMsg := message
	participantType := normalizeRoomSenderType(message.SenderType)
	event := ClawRoomCollaborationEvent{
		Type:            WSTypeMessageUpdated,
		Message:         &copyMsg,
		MessageID:       message.ID,
		ParticipantType: participantType,
		ParticipantID:   message.SenderID,
	}
	if participantType == "worker" && message.SenderID != "" {
		event.SessionUser = roomScopedSessionUser(room, ContainerEntry{Name: message.SenderID})
	}
	return event
}

func surfaceEventCollaborationEvent(participantType, participantID string, payload map[string]any) ClawRoomCollaborationEvent {
	payloadCopy := make(map[string]any, len(payload))
	for key, value := range payload {
		payloadCopy[key] = value
	}
	return ClawRoomCollaborationEvent{
		Type:            WSTypeSurfaceEvent,
		ParticipantType: participantType,
		ParticipantID:   participantID,
		Payload:         payloadCopy,
	}
}

func workerStreamChunkCollaborationEvent(
	room ClawRoomEntry,
	worker ContainerEntry,
	messageID, chunk string,
	done bool,
) ClawRoomCollaborationEvent {
	event := ClawRoomCollaborationEvent{
		Type:            WSTypeMessageChunk,
		MessageID:       messageID,
		Chunk:           chunk,
		ParticipantType: "worker",
		ParticipantID:   worker.Name,
		SessionUser:     roomScopedSessionUser(room, worker),
	}
	if done {
		event.Status = "complete"
	} else {
		event.Status = "streaming"
	}
	return event
}

func collaborationEventToWSOutbound(event ClawRoomCollaborationEvent) WSOutboundMessage {
	outbound := WSOutboundMessage{
		Type:            event.Type,
		RoomID:          event.RoomID,
		MessageID:       event.MessageID,
		Chunk:           event.Chunk,
		Status:          event.Status,
		Error:           event.Error,
		Timestamp:       event.Timestamp,
		ParticipantType: event.ParticipantType,
		ParticipantID:   event.ParticipantID,
		SessionUser:     event.SessionUser,
		Payload:         event.Payload,
	}
	if event.Message != nil {
		copyMsg := *event.Message
		outbound.Message = &copyMsg
	}
	return outbound
}

func collaborationEventToSSE(event ClawRoomCollaborationEvent) clawRoomStreamEvent {
	sseType := event.Type
	if sseType == WSTypeNewMessage {
		sseType = "message"
	}
	sseEvent := clawRoomStreamEvent{
		Type:            sseType,
		RoomID:          event.RoomID,
		MessageID:       event.MessageID,
		Chunk:           event.Chunk,
		Status:          event.Status,
		ParticipantType: event.ParticipantType,
		ParticipantID:   event.ParticipantID,
		SessionUser:     event.SessionUser,
		Payload:         event.Payload,
	}
	if event.Message != nil {
		copyMsg := *event.Message
		sseEvent.Message = &copyMsg
	}
	return sseEvent
}

func shouldCacheCollaborationEvent(event ClawRoomCollaborationEvent) bool {
	return event.Message != nil && event.Type == WSTypeNewMessage
}

func (h *OpenClawHandler) roomWSClientMap(roomID string) *sync.Map {
	if existing, ok := h.roomWSClients.Load(roomID); ok {
		return existing.(*sync.Map)
	}
	clients := &sync.Map{}
	actual, _ := h.roomWSClients.LoadOrStore(roomID, clients)
	return actual.(*sync.Map)
}

func (h *OpenClawHandler) roomSSEClientMap(roomID string) *sync.Map {
	if existing, ok := h.roomSSEClients.Load(roomID); ok {
		return existing.(*sync.Map)
	}
	clients := &sync.Map{}
	actual, _ := h.roomSSEClients.LoadOrStore(roomID, clients)
	return actual.(*sync.Map)
}

// publishRoomCollaborationEvent is the single fan-out entry for room live events.
func (h *OpenClawHandler) publishRoomCollaborationEvent(roomID string, event ClawRoomCollaborationEvent) {
	event.RoomID = roomID
	if event.Timestamp == "" {
		event.Timestamp = time.Now().UTC().Format(time.RFC3339)
	}
	event = cloneCollaborationEvent(event)

	if shouldCacheCollaborationEvent(event) {
		h.roomSSELastEvent.Store(roomID, collaborationEventToSSE(event))
	}

	h.fanOutCollaborationToWebSocket(roomID, event)
	h.fanOutCollaborationToSSE(roomID, event)
}

func (h *OpenClawHandler) fanOutCollaborationToWebSocket(roomID string, event ClawRoomCollaborationEvent) {
	outbound := collaborationEventToWSOutbound(event)
	clients := h.roomWSClientMap(roomID)
	clients.Range(func(_, value any) bool {
		client, ok := value.(*WSClient)
		if !ok {
			return true
		}

		switch client.enqueue(outbound) {
		case wsEnqueueFull:
			log.Printf("openclaw: WS client %s buffer full, skipping event", client.clientID)
		case wsEnqueueClosed:
			return true
		}
		return true
	})
}

func (h *OpenClawHandler) fanOutCollaborationToSSE(roomID string, event ClawRoomCollaborationEvent) {
	sseEvent := collaborationEventToSSE(event)
	clients := h.roomSSEClientMap(roomID)
	clients.Range(func(_, value any) bool {
		ch, ok := value.(chan clawRoomStreamEvent)
		if !ok {
			return true
		}
		select {
		case ch <- cloneRoomEventWithMessage(sseEvent):
		default:
		}
		return true
	})
}
