package handlers

import (
	"context"
	"errors"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/jsonunicode"
)

// WebSocket message types for ClawRoom.
// WebSocket is the primary room collaboration transport; SSE (/stream) is a legacy fallback.
const (
	WSTypeSendMessage     = "send_message"
	WSTypeSurfaceEvent    = "surface_event"
	WSTypePing            = "ping"
	WSTypePong            = "pong"
	WSTypeNewMessage      = "new_message"
	WSTypeMessageUpdated  = "message_updated"
	WSTypeMessageChunk    = "message_chunk"
	WSTypeToolTraceUpdate = "tool_trace_update"
	WSTypeConnected       = "connected"
	WSTypeError           = "error"

	roomWSPingInterval = 5 * time.Minute
	roomWSWriteTimeout = 5 * time.Minute
	roomWSReadTimeout  = roomWSPingInterval * 2
)

// WSInboundMessage represents a message from client to server
type WSInboundMessage struct {
	Type       string         `json:"type"`
	Content    string         `json:"content,omitempty"`
	SenderType string         `json:"senderType,omitempty"`
	SenderID   string         `json:"senderId,omitempty"`
	SenderName string         `json:"senderName,omitempty"`
	Mentions   []string       `json:"mentions,omitempty"`
	Payload    map[string]any `json:"payload,omitempty"`
}

// WSOutboundMessage represents a message from server to client
type WSOutboundMessage struct {
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

// WSClient represents a WebSocket client connection
type WSClient struct {
	conn     *websocket.Conn
	send     chan WSOutboundMessage
	done     chan struct{}
	ctx      context.Context
	roomID   string
	clientID string
	handler  *OpenClawHandler
	closed   bool
	closeMu  sync.Mutex
}

type wsEnqueueResult uint8

const (
	wsEnqueueSent wsEnqueueResult = iota
	wsEnqueueClosed
	wsEnqueueFull
)

var wsUpgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin:     auth.ValidWebSocketOrigin,
}

func wsOutboundFromLastRoomEvent(roomID string, event clawRoomStreamEvent) (WSOutboundMessage, bool) {
	if event.Message == nil {
		return WSOutboundMessage{}, false
	}
	copyMsg := *event.Message
	return WSOutboundMessage{
		Type:    WSTypeNewMessage,
		RoomID:  roomID,
		Message: &copyMsg,
	}, true
}

func (h *OpenClawHandler) replayLastRoomEventToClient(client *WSClient, roomID string) {
	lastAny, ok := h.roomSSELastEvent.Load(roomID)
	if !ok {
		return
	}
	lastEvent, ok := lastAny.(clawRoomStreamEvent)
	if !ok {
		return
	}
	replay, ok := wsOutboundFromLastRoomEvent(roomID, lastEvent)
	if !ok {
		return
	}
	switch client.enqueue(replay) {
	case wsEnqueueFull:
		log.Printf("openclaw: WS client %s buffer full, skipping room replay", client.clientID)
	case wsEnqueueClosed:
		return
	}
}

// handleRoomWebSocket handles WebSocket connections for a room
func (h *OpenClawHandler) handleRoomWebSocket(w http.ResponseWriter, r *http.Request, roomID string) {
	// Verify room exists
	h.mu.RLock()
	rooms, err := h.loadRooms()
	h.mu.RUnlock()
	if err != nil {
		writeJSONError(w, "Failed to load rooms", http.StatusInternalServerError)
		return
	}
	if findRoomByID(rooms, roomID) == nil {
		writeJSONError(w, "room not found", http.StatusNotFound)
		return
	}

	// Upgrade to WebSocket
	conn, err := wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("openclaw: WebSocket upgrade failed: %v", err)
		return
	}

	clientID := generateRoomEntityID("ws-client")
	client := &WSClient{
		conn:     conn,
		send:     make(chan WSOutboundMessage, 128),
		done:     make(chan struct{}),
		ctx:      r.Context(),
		roomID:   roomID,
		clientID: clientID,
		handler:  h,
	}

	// Register client
	clients := h.roomWSClientMap(roomID)
	clients.Store(clientID, client)

	log.Printf("openclaw: WebSocket client %s connected to room %s", clientID, roomID)

	// Send connected message
	if client.enqueue(WSOutboundMessage{
		Type:   WSTypeConnected,
		RoomID: roomID,
	}) != wsEnqueueSent {
		client.close()
		return
	}
	h.replayLastRoomEventToClient(client, roomID)

	// Start read/write goroutines
	go client.writePump()
	go client.readPump()

	// Keep ServeHTTP (and therefore the auth middleware's live session watch)
	// alive for the lifetime of this hijacked connection.
	<-client.done
}

// writePump handles writing messages to the WebSocket connection
func (c *WSClient) writePump() {
	ticker := time.NewTicker(roomWSPingInterval)
	defer func() {
		ticker.Stop()
		c.close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			if !ok {
				// Channel closed
				_ = c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			_ = c.conn.SetWriteDeadline(time.Now().Add(roomWSWriteTimeout))
			if err := c.conn.WriteJSON(message); err != nil {
				log.Printf("openclaw: WS write error for client %s: %v", c.clientID, err)
				return
			}

		case <-ticker.C:
			_ = c.conn.SetWriteDeadline(time.Now().Add(roomWSWriteTimeout))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		case <-c.ctx.Done():
			_ = c.conn.SetWriteDeadline(time.Now().Add(roomWSWriteTimeout))
			_ = c.conn.WriteControl(
				websocket.CloseMessage,
				websocket.FormatCloseMessage(websocket.ClosePolicyViolation, "authorization expired or revoked"),
				time.Now().Add(time.Second),
			)
			return
		}
	}
}

// readPump handles reading messages from the WebSocket connection
func (c *WSClient) readPump() {
	defer func() {
		c.close()
	}()

	c.conn.SetReadLimit(64 * 1024) // 64KB max message size
	_ = c.conn.SetReadDeadline(time.Now().Add(roomWSReadTimeout))
	c.conn.SetPongHandler(func(string) error {
		_ = c.conn.SetReadDeadline(time.Now().Add(roomWSReadTimeout))
		return nil
	})

	for {
		_, data, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure, websocket.CloseNoStatusReceived) {
				log.Printf("openclaw: WS read error for client %s: %v", c.clientID, err)
			}
			// Close 1005 (NoStatusReceived) is normal when client navigates away or switches rooms
			return
		}

		var msg WSInboundMessage
		if !jsonunicode.Valid(data) {
			c.sendError("invalid message Unicode")
			continue
		}
		if _, err := decodeStrictJSONBytes(data, &msg); err != nil {
			c.sendError("invalid message format")
			continue
		}

		c.handleMessage(msg)
	}
}

// handleMessage processes an inbound WebSocket message
func (c *WSClient) handleMessage(msg WSInboundMessage) {
	switch msg.Type {
	case WSTypePing:
		_ = c.enqueue(WSOutboundMessage{Type: WSTypePong})

	case WSTypeSendMessage:
		c.handleSendMessage(msg)

	case WSTypeSurfaceEvent:
		c.handleSurfaceEvent(msg)

	default:
		c.sendError("unknown message type: " + msg.Type)
	}
}

// handleSurfaceEvent publishes embedded OpenClaw surface activity to the room bus.
func (c *WSClient) handleSurfaceEvent(msg WSInboundMessage) {
	if msg.Payload == nil {
		c.sendError("payload is required")
		return
	}

	actor, browserActor, allowed := c.authorizeMutation()
	if !allowed {
		return
	}

	participantType := normalizeRoomSenderType(msg.SenderType)
	if participantType != "user" && participantType != "leader" && participantType != "worker" && participantType != "system" {
		participantType = "user"
	}
	participantID := msg.SenderID
	if browserActor {
		participantType = "user"
		participantID = actor.UserID
	}
	if len([]byte(strings.TrimSpace(participantID))) > maximumOpenClawSenderFieldBytes ||
		len([]byte(strings.TrimSpace(msg.SenderName))) > maximumOpenClawSenderFieldBytes {
		c.sendError("sender identity is too large")
		return
	}
	if participantID == "" {
		switch participantType {
		case "user":
			participantID = "playground-user"
		case "leader", "worker":
			participantID = sanitizeContainerName(msg.SenderName)
		case "system":
			participantID = "clawos-system"
		}
	}

	c.handler.publishRoomCollaborationEvent(
		c.roomID,
		surfaceEventCollaborationEvent(participantType, participantID, msg.Payload),
	)
}

// handleSendMessage handles sending a new message to the room
func (c *WSClient) handleSendMessage(msg WSInboundMessage) {
	if !c.handler.canSendRoomMessages() {
		c.sendError("read-only mode enabled")
		return
	}
	actor, browserActor, allowed := c.authorizeMutation()
	if !allowed {
		return
	}

	content := msg.Content
	if content == "" {
		c.sendError("content is required")
		return
	}
	if len([]byte(strings.TrimSpace(content))) > maximumOpenClawRoomMessageBytes {
		c.sendError("content is too large")
		return
	}

	// Load room
	c.handler.mu.RLock()
	rooms, err := c.handler.loadRooms()
	c.handler.mu.RUnlock()
	if err != nil {
		c.sendError("failed to load rooms")
		return
	}

	room := findRoomByID(rooms, c.roomID)
	if room == nil {
		c.sendError("room not found")
		return
	}

	// Determine sender info
	senderType := normalizeRoomSenderType(msg.SenderType)
	if senderType != "user" && senderType != "leader" && senderType != "worker" && senderType != "system" {
		senderType = "user"
	}

	senderName := msg.SenderName
	if senderName == "" {
		switch senderType {
		case "leader":
			senderName = "Leader"
		case "worker":
			senderName = "Worker"
		case "system":
			senderName = "System"
		default:
			senderName = "You"
		}
	}

	senderID := msg.SenderID
	if senderID == "" {
		switch senderType {
		case "user":
			senderID = "playground-user"
		case "leader", "worker":
			senderID = sanitizeContainerName(senderName)
		}
	}
	if browserActor {
		senderType = "user"
		senderID = actor.UserID
		senderName = strings.TrimSpace(actor.Name)
		if senderName == "" {
			senderName = actor.Email
		}
	}
	if len([]byte(strings.TrimSpace(senderID))) > maximumOpenClawSenderFieldBytes ||
		len([]byte(strings.TrimSpace(senderName))) > maximumOpenClawSenderFieldBytes {
		c.sendError("sender identity is too large")
		return
	}

	// Create and save message
	created := newRoomMessage(*room, senderType, senderID, senderName, content, nil)

	if err := c.handler.appendRoomMessageWS(room.ID, created); err != nil {
		if errors.Is(err, workflowstore.ErrOpenClawRoomMessageLimit) {
			c.sendError("room message limit reached")
			return
		}
		c.sendError("failed to save message")
		return
	}

	// Trigger automation for non-system messages
	if senderType != "system" {
		c.handler.enqueueRoomAutomation(room.ID, created.ID)
	}
}

func (c *WSClient) authorizeMutation() (auth.AuthContext, bool, bool) {
	if !auth.HasLiveAuthorization(c.ctx) {
		// OpenClaw's internal MCP/automation handler composition intentionally
		// has no browser auth state and supplies its own trusted worker identity.
		return auth.AuthContext{}, false, true
	}
	actor, err := auth.RevalidateAuthorization(c.ctx, auth.PermOpenClawUse)
	if err == nil {
		return actor, true, true
	}
	if errors.Is(err, auth.ErrLivePermissionDenied) {
		c.sendError("forbidden: openclaw.use is required")
		return auth.AuthContext{}, true, false
	}
	_ = c.conn.WriteControl(
		websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.ClosePolicyViolation, "authorization expired or revoked"),
		time.Now().Add(time.Second),
	)
	c.close()
	return auth.AuthContext{}, true, false
}

// appendRoomMessageWS appends a message and broadcasts via WebSocket
func (h *OpenClawHandler) appendRoomMessageWS(roomID string, message ClawRoomMessage) error {
	return h.appendRoomMessage(roomID, message)
}

// sendError sends an error message to the client
func (c *WSClient) sendError(errMsg string) {
	_ = c.enqueue(WSOutboundMessage{
		Type:  WSTypeError,
		Error: errMsg,
	})
}

// enqueue is the only producer-side access to send. Holding closeMu through the
// non-blocking send makes channel closure and every producer mutually exclusive.
func (c *WSClient) enqueue(message WSOutboundMessage) wsEnqueueResult {
	c.closeMu.Lock()
	defer c.closeMu.Unlock()

	if c.closed {
		return wsEnqueueClosed
	}

	select {
	case c.send <- message:
		return wsEnqueueSent
	default:
		return wsEnqueueFull
	}
}

// close cleans up the client connection
func (c *WSClient) close() {
	c.closeMu.Lock()
	defer c.closeMu.Unlock()

	if c.closed {
		return
	}
	c.closed = true

	// Unregister from room
	clients := c.handler.roomWSClientMap(c.roomID)
	clients.Delete(c.clientID)

	// Close connection and channel
	_ = c.conn.Close()
	close(c.send)
	close(c.done)

	log.Printf("openclaw: WebSocket client %s disconnected from room %s", c.clientID, c.roomID)
}
