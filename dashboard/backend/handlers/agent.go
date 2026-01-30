package handlers

import (
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

// AgentHandler handles proxying requests to the Python agent service
type AgentHandler struct {
	agentServiceURL string
	wsDialer        *websocket.Dialer
}

// NewAgentHandler creates a new agent handler
func NewAgentHandler(agentServiceURL string) *AgentHandler {
	return &AgentHandler{
		agentServiceURL: agentServiceURL,
		wsDialer: &websocket.Dialer{
			HandshakeTimeout: 10 * time.Second,
		},
	}
}

// WebSocketHandler proxies WebSocket connections to the agent service
func (h *AgentHandler) WebSocketHandler() http.HandlerFunc {
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true // Allow all origins
		},
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
	}

	return func(w http.ResponseWriter, r *http.Request) {
		// Parse the agent service URL
		targetURL, err := url.Parse(h.agentServiceURL)
		if err != nil {
			log.Printf("Failed to parse agent service URL: %v", err)
			http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		}

		// Build WebSocket URL
		wsScheme := "ws"
		if targetURL.Scheme == "https" {
			wsScheme = "wss"
		}
		wsURL := wsScheme + "://" + targetURL.Host + "/ws"

		log.Printf("Proxying WebSocket connection to: %s", wsURL)

		// Connect to backend WebSocket
		backendConn, resp, err := h.wsDialer.Dial(wsURL, nil)
		if err != nil {
			log.Printf("Failed to connect to agent service WebSocket: %v", err)
			if resp != nil {
				log.Printf("Response status: %d", resp.StatusCode)
			}
			http.Error(w, "Failed to connect to agent service", http.StatusBadGateway)
			return
		}
		defer backendConn.Close()

		// Upgrade client connection to WebSocket
		clientConn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("Failed to upgrade client connection: %v", err)
			return
		}
		defer clientConn.Close()

		log.Printf("WebSocket proxy established")

		// Create error channel
		errChan := make(chan error, 2)

		// Proxy client -> backend
		go func() {
			for {
				messageType, message, err := clientConn.ReadMessage()
				if err != nil {
					errChan <- err
					return
				}
				if err := backendConn.WriteMessage(messageType, message); err != nil {
					errChan <- err
					return
				}
			}
		}()

		// Proxy backend -> client
		go func() {
			for {
				messageType, message, err := backendConn.ReadMessage()
				if err != nil {
					errChan <- err
					return
				}
				if err := clientConn.WriteMessage(messageType, message); err != nil {
					errChan <- err
					return
				}
			}
		}()

		// Wait for error from either direction
		<-errChan
		log.Printf("WebSocket proxy closed")
	}
}

// RESTProxyHandler returns a handler that proxies REST requests to the agent service
func (h *AgentHandler) RESTProxyHandler() http.HandlerFunc {
	targetURL, err := url.Parse(h.agentServiceURL)
	if err != nil {
		log.Printf("Failed to parse agent service URL: %v", err)
		return func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "Agent service not configured", http.StatusInternalServerError)
		}
	}

	proxy := httputil.NewSingleHostReverseProxy(targetURL)

	// Custom director to rewrite paths
	originalDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		originalDirector(req)
		// Strip /api/agent prefix and forward to /api
		req.URL.Path = strings.TrimPrefix(req.URL.Path, "/api/agent")
		if !strings.HasPrefix(req.URL.Path, "/") {
			req.URL.Path = "/" + req.URL.Path
		}
		req.URL.RawPath = ""
		req.Host = targetURL.Host
		log.Printf("Proxying REST request to agent service: %s %s", req.Method, req.URL.Path)
	}

	// Custom error handler
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("Agent service proxy error: %v", err)
		w.Header().Set("Content-Type", "application/json")
		http.Error(w, `{"error":"Agent service unavailable","message":"`+err.Error()+`"}`, http.StatusBadGateway)
	}

	// Custom response modifier for CORS
	proxy.ModifyResponse = func(resp *http.Response) error {
		resp.Header.Set("Access-Control-Allow-Origin", "*")
		resp.Header.Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		resp.Header.Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		return nil
	}

	return func(w http.ResponseWriter, r *http.Request) {
		proxy.ServeHTTP(w, r)
	}
}

// HealthHandler checks the health of the agent service
func (h *AgentHandler) HealthHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Make request to agent service health endpoint
		resp, err := http.Get(h.agentServiceURL + "/health")
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte(`{"status":"unhealthy","error":"` + err.Error() + `"}`))
			return
		}
		defer resp.Body.Close()

		// Copy response
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
	}
}

// ModelsHandler returns available agent models
func (h *AgentHandler) ModelsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Make request to agent service models endpoint
		resp, err := http.Get(h.agentServiceURL + "/api/models")
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte(`{"error":"` + err.Error() + `"}`))
			return
		}
		defer resp.Body.Close()

		// Copy response
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
	}
}
