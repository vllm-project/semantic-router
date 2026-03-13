package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// WebSocket message types for ClawRoom
const (
	WSTypeSendMessage    = "send_message"
	WSTypePing           = "ping"
	WSTypePong           = "pong"
	WSTypeNewMessage     = "new_message"
	WSTypeMessageUpdated = "message_updated"
	WSTypeMessageChunk   = "message_chunk"
	WSTypeConnected      = "connected"
	WSTypeError          = "error"

	roomWSPingInterval = 5 * time.Minute
	roomWSWriteTimeout = 5 * time.Minute
	roomWSReadTimeout  = roomWSPingInterval * 2
)

// WSInboundMessage represents a message from client to server
type WSInboundMessage struct {
	Type       string   `json:"type"`
	Content    string   `json:"content,omitempty"`
	SenderType string   `json:"senderType,omitempty"`
	SenderID   string   `json:"senderId,omitempty"`
	SenderName string   `json:"senderName,omitempty"`
	Mentions   []string `json:"mentions,omitempty"`
}

// WSOutboundMessage represents a message from server to client
type WSOutboundMessage struct {
	Type      string           `json:"type"`
	RoomID    string           `json:"roomId,omitempty"`
	Message   *ClawRoomMessage `json:"message,omitempty"`
	MessageID string           `json:"messageId,omitempty"`
	Chunk     string           `json:"chunk,omitempty"`
	Status    string           `json:"status,omitempty"`
	Error     string           `json:"error,omitempty"`
	Timestamp string           `json:"timestamp,omitempty"`
}

// WSClient represents a WebSocket client connection
type WSClient struct {
	conn     *websocket.Conn
	send     chan WSOutboundMessage
	roomID   string
	clientID string
	handler  *OpenClawHandler
	closed   bool
	closeMu  sync.Mutex
}

var wsUpgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for now
	},
}

// roomWSClients stores WebSocket clients per room
// Key: roomID, Value: *sync.Map (clientID -> *WSClient)
func (h *OpenClawHandler) roomWSClientMap(roomID string) *sync.Map {
	if existing, ok := h.roomSSEClients.Load(roomID); ok {
		return existing.(*sync.Map)
	}
	clients := &sync.Map{}
	actual, _ := h.roomSSEClients.LoadOrStore(roomID, clients)
	return actual.(*sync.Map)
}

// publishRoomWSEvent broadcasts an event to all WebSocket clients in a room
func (h *OpenClawHandler) publishRoomWSEvent(roomID string, event WSOutboundMessage) {
	event.RoomID = roomID
	event.Timestamp = time.Now().UTC().Format(time.RFC3339)

	clients := h.roomWSClientMap(roomID)
	clients.Range(func(_, value any) bool {
		client, ok := value.(*WSClient)
		if !ok {
			// Backward compatibility: might be SSE channel
			if ch, ok := value.(chan clawRoomStreamEvent); ok {
				// Convert to SSE event
				sseEvent := clawRoomStreamEvent{
					Type:   event.Type,
					RoomID: roomID,
				}
				if event.Message != nil {
					sseEvent.Message = event.Message
				}
				select {
				case ch <- sseEvent:
				default:
				}
			}
			return true
		}

		client.closeMu.Lock()
		if client.closed {
			client.closeMu.Unlock()
			return true
		}
		client.closeMu.Unlock()

		select {
		case client.send <- event:
		default:
			// Client buffer full, skip
			log.Printf("openclaw: WS client %s buffer full, skipping event", client.clientID)
		}
		return true
	})
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
		send:     make(chan WSOutboundMessage, 32),
		roomID:   roomID,
		clientID: clientID,
		handler:  h,
	}

	// Register client
	clients := h.roomWSClientMap(roomID)
	clients.Store(clientID, client)

	log.Printf("openclaw: WebSocket client %s connected to room %s", clientID, roomID)

	// Send connected message
	client.send <- WSOutboundMessage{
		Type:   WSTypeConnected,
		RoomID: roomID,
	}

	// Start read/write goroutines
	go client.writePump()
	go client.readPump()
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
		if err := json.Unmarshal(data, &msg); err != nil {
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
		c.send <- WSOutboundMessage{Type: WSTypePong}

	case WSTypeSendMessage:
		c.handleSendMessage(msg)

	default:
		c.sendError("unknown message type: " + msg.Type)
	}
}

// handleSendMessage handles sending a new message to the room
func (c *WSClient) handleSendMessage(msg WSInboundMessage) {
	if !c.handler.canSendRoomMessages() {
		c.sendError("read-only mode enabled")
		return
	}

	content := msg.Content
	if content == "" {
		c.sendError("content is required")
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

	// Create and save message
	created := newRoomMessage(*room, senderType, senderID, senderName, content, nil)

	if err := c.handler.appendRoomMessageWS(room.ID, created); err != nil {
		c.sendError("failed to save message: " + err.Error())
		return
	}

	// Trigger automation for non-system messages
	if senderType != "system" {
		go c.handler.processRoomUserMessage(room.ID, created.ID)
	}
}

// appendRoomMessageWS appends a message and broadcasts via WebSocket
func (h *OpenClawHandler) appendRoomMessageWS(roomID string, message ClawRoomMessage) error {
	h.mu.Lock()
	messages, err := h.loadRoomMessages(roomID)
	if err != nil {
		h.mu.Unlock()
		return err
	}
	messages = append(messages, message)
	if err := h.saveRoomMessages(roomID, messages); err != nil {
		h.mu.Unlock()
		return err
	}
	h.mu.Unlock()

	// Broadcast via WebSocket
	h.publishRoomWSEvent(roomID, WSOutboundMessage{
		Type:    WSTypeNewMessage,
		Message: &message,
	})

	// Also publish to SSE for backward compatibility
	h.roomSSELastEvent.Store(roomID, clawRoomStreamEvent{
		Type:    "message",
		RoomID:  roomID,
		Message: &message,
	})

	return nil
}

// sendError sends an error message to the client
func (c *WSClient) sendError(errMsg string) {
	c.send <- WSOutboundMessage{
		Type:  WSTypeError,
		Error: errMsg,
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

	log.Printf("openclaw: WebSocket client %s disconnected from room %s", c.clientID, c.roomID)
}
