package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestWorkerByIDHandler_SetLeaderRoleUpdatesTeamAndDemotesOthers(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	if err := h.saveTeams([]TeamEntry{
		{
			ID:        "core",
			Name:      "Core Team",
			LeaderID:  "worker-a",
			CreatedAt: "2026-01-01T00:00:00Z",
			UpdatedAt: "2026-01-01T00:00:00Z",
		},
	}); err != nil {
		t.Fatalf("failed to seed teams: %v", err)
	}
	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:     "worker-a",
			Port:     18788,
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "token",
			DataDir:  tempDir,
			TeamID:   "core",
			TeamName: "Core Team",
			RoleKind: "leader",
		},
		{
			Name:     "worker-b",
			Port:     18789,
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "token",
			DataDir:  tempDir,
			TeamID:   "core",
			TeamName: "Core Team",
			RoleKind: "worker",
		},
	}); err != nil {
		t.Fatalf("failed to seed workers: %v", err)
	}

	updateReq := httptest.NewRequest(http.MethodPut, "/api/openclaw/workers/worker-b", strings.NewReader(`{
		"roleKind":"leader"
	}`))
	updateResp := httptest.NewRecorder()
	h.WorkerByIDHandler().ServeHTTP(updateResp, updateReq)
	if updateResp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", updateResp.Code, updateResp.Body.String())
	}

	teams, err := h.loadTeams()
	if err != nil {
		t.Fatalf("failed to load teams: %v", err)
	}
	if len(teams) != 1 || teams[0].LeaderID != "worker-b" {
		t.Fatalf("expected team leader to become worker-b, got %+v", teams)
	}

	entries, err := h.loadRegistry()
	if err != nil {
		t.Fatalf("failed to load workers: %v", err)
	}
	roleByName := map[string]string{}
	leaderRole := ""
	leaderVibe := ""
	leaderPrinciples := ""
	for _, entry := range entries {
		roleByName[entry.Name] = normalizeRoleKind(entry.RoleKind)
		if entry.Name == "worker-b" {
			leaderRole = strings.TrimSpace(entry.AgentRole)
			leaderVibe = strings.TrimSpace(entry.AgentVibe)
			leaderPrinciples = strings.TrimSpace(entry.AgentPrinciples)
		}
	}
	if roleByName["worker-b"] != "leader" {
		t.Fatalf("worker-b should be leader, got %q", roleByName["worker-b"])
	}
	if roleByName["worker-a"] != "worker" {
		t.Fatalf("worker-a should be demoted to worker, got %q", roleByName["worker-a"])
	}
	if leaderRole == "" || leaderVibe == "" || leaderPrinciples == "" {
		t.Fatalf("leader metadata defaults should be populated, got role=%q vibe=%q principles=%q", leaderRole, leaderVibe, leaderPrinciples)
	}
}

func TestRoomsHandler_CreateAndMessageFlowWithoutAutomation(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	if err := h.saveTeams([]TeamEntry{{
		ID:        "team-a",
		Name:      "Team A",
		CreatedAt: "2026-01-01T00:00:00Z",
		UpdatedAt: "2026-01-01T00:00:00Z",
	}}); err != nil {
		t.Fatalf("failed to seed team: %v", err)
	}

	createRoomReq := httptest.NewRequest(http.MethodPost, "/api/openclaw/rooms", strings.NewReader(`{
		"teamId":"team-a",
		"name":"Planning"
	}`))
	createRoomResp := httptest.NewRecorder()
	h.RoomsHandler().ServeHTTP(createRoomResp, createRoomReq)
	if createRoomResp.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", createRoomResp.Code, createRoomResp.Body.String())
	}
	var room ClawRoomEntry
	if err := json.Unmarshal(createRoomResp.Body.Bytes(), &room); err != nil {
		t.Fatalf("failed to parse room create response: %v", err)
	}
	if room.ID == "" {
		t.Fatalf("room id should not be empty")
	}

	listReq := httptest.NewRequest(http.MethodGet, "/api/openclaw/rooms?teamId=team-a", nil)
	listResp := httptest.NewRecorder()
	h.RoomsHandler().ServeHTTP(listResp, listReq)
	if listResp.Code != http.StatusOK {
		t.Fatalf("expected 200 for room list, got %d", listResp.Code)
	}

	postReq := httptest.NewRequest(http.MethodPost, fmt.Sprintf("/api/openclaw/rooms/%s/messages", room.ID), strings.NewReader(`{
		"senderType":"system",
		"senderName":"test",
		"content":"hello @leader"
	}`))
	postResp := httptest.NewRecorder()
	h.RoomByIDHandler().ServeHTTP(postResp, postReq)
	if postResp.Code != http.StatusCreated {
		t.Fatalf("expected 201 for message post, got %d: %s", postResp.Code, postResp.Body.String())
	}

	msgListReq := httptest.NewRequest(http.MethodGet, fmt.Sprintf("/api/openclaw/rooms/%s/messages?limit=10", room.ID), nil)
	msgListResp := httptest.NewRecorder()
	h.RoomByIDHandler().ServeHTTP(msgListResp, msgListReq)
	if msgListResp.Code != http.StatusOK {
		t.Fatalf("expected 200 for message list, got %d", msgListResp.Code)
	}
	var messages []ClawRoomMessage
	if err := json.Unmarshal(msgListResp.Body.Bytes(), &messages); err != nil {
		t.Fatalf("failed to parse message list: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if len(messages[0].Mentions) != 1 || messages[0].Mentions[0] != "leader" {
		t.Fatalf("mention parsing mismatch: %+v", messages[0].Mentions)
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, fmt.Sprintf("/api/openclaw/rooms/%s", room.ID), nil)
	deleteResp := httptest.NewRecorder()
	h.RoomByIDHandler().ServeHTTP(deleteResp, deleteReq)
	if deleteResp.Code != http.StatusOK {
		t.Fatalf("expected 200 for room delete, got %d: %s", deleteResp.Code, deleteResp.Body.String())
	}

	getRoomReq := httptest.NewRequest(http.MethodGet, fmt.Sprintf("/api/openclaw/rooms/%s", room.ID), nil)
	getRoomResp := httptest.NewRecorder()
	h.RoomByIDHandler().ServeHTTP(getRoomResp, getRoomReq)
	if getRoomResp.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for deleted room, got %d", getRoomResp.Code)
	}
}

func TestOpenClawReadonlyBlocksManagementMutations(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), true)

	testCases := []struct {
		name    string
		method  string
		path    string
		body    string
		handler http.HandlerFunc
	}{
		{name: "team create", method: http.MethodPost, path: "/api/openclaw/teams", body: `{}`, handler: h.TeamsHandler()},
		{name: "team update", method: http.MethodPut, path: "/api/openclaw/teams/team-a", body: `{}`, handler: h.TeamByIDHandler()},
		{name: "team delete", method: http.MethodDelete, path: "/api/openclaw/teams/team-a", handler: h.TeamByIDHandler()},
		{name: "worker create", method: http.MethodPost, path: "/api/openclaw/workers", body: `{}`, handler: h.WorkersHandler()},
		{name: "worker update", method: http.MethodPut, path: "/api/openclaw/workers/worker-a", body: `{}`, handler: h.WorkerByIDHandler()},
		{name: "worker delete", method: http.MethodDelete, path: "/api/openclaw/workers/worker-a", handler: h.WorkerByIDHandler()},
		{name: "room create", method: http.MethodPost, path: "/api/openclaw/rooms", body: `{}`, handler: h.RoomsHandler()},
		{name: "room delete", method: http.MethodDelete, path: "/api/openclaw/rooms/room-a", handler: h.RoomByIDHandler()},
		{name: "start", method: http.MethodPost, path: "/api/openclaw/start", body: `{}`, handler: h.StartHandler()},
		{name: "stop", method: http.MethodPost, path: "/api/openclaw/stop", body: `{}`, handler: h.StopHandler()},
		{name: "remove", method: http.MethodDelete, path: "/api/openclaw/containers/worker-a", handler: h.DeleteHandler()},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var body *strings.Reader
			if tc.body != "" {
				body = strings.NewReader(tc.body)
			} else {
				body = strings.NewReader("")
			}
			req := httptest.NewRequest(tc.method, tc.path, body)
			resp := httptest.NewRecorder()
			tc.handler.ServeHTTP(resp, req)
			if resp.Code != http.StatusForbidden {
				t.Fatalf("expected 403, got %d: %s", resp.Code, resp.Body.String())
			}
			if !strings.Contains(resp.Body.String(), "Read-only mode enabled") {
				t.Fatalf("expected readonly error, got %q", resp.Body.String())
			}
		})
	}
}

func TestRoomsHandler_CreateRoomAutoSuffixAvoidsConflict(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	if err := h.saveTeams([]TeamEntry{{
		ID:        "llm-router-lab",
		Name:      "LLM Router Lab",
		CreatedAt: "2026-01-01T00:00:00Z",
		UpdatedAt: "2026-01-01T00:00:00Z",
	}}); err != nil {
		t.Fatalf("failed to seed team: %v", err)
	}

	createWithName := func(name string) ClawRoomEntry {
		req := httptest.NewRequest(http.MethodPost, "/api/openclaw/rooms", strings.NewReader(fmt.Sprintf(`{
			"teamId":"llm-router-lab",
			"name":"%s"
		}`, name)))
		resp := httptest.NewRecorder()
		h.RoomsHandler().ServeHTTP(resp, req)
		if resp.Code != http.StatusCreated {
			t.Fatalf("expected 201, got %d: %s", resp.Code, resp.Body.String())
		}
		var created ClawRoomEntry
		if err := json.Unmarshal(resp.Body.Bytes(), &created); err != nil {
			t.Fatalf("failed to parse room create response: %v", err)
		}
		return created
	}

	first := createWithName("llm-router-lab-room")
	second := createWithName("llm-router-lab-room")
	if first.ID == second.ID {
		t.Fatalf("room IDs should be unique, got duplicated id %q", first.ID)
	}

	baseID := sanitizeRoomID("llm-router-lab-room")
	for _, room := range []ClawRoomEntry{first, second} {
		if !strings.HasPrefix(room.ID, baseID+"-") {
			t.Fatalf("room ID %q should keep fixed base prefix %q", room.ID, baseID+"-")
		}
		suffix := strings.TrimPrefix(room.ID, baseID+"-")
		if len(suffix) < 4 {
			t.Fatalf("room ID %q should include a short dynamic suffix", room.ID)
		}
	}

	duplicateIDReq := httptest.NewRequest(http.MethodPost, "/api/openclaw/rooms", strings.NewReader(fmt.Sprintf(`{
		"teamId":"llm-router-lab",
		"name":"manual",
		"id":"%s"
	}`, first.ID)))
	duplicateIDResp := httptest.NewRecorder()
	h.RoomsHandler().ServeHTTP(duplicateIDResp, duplicateIDReq)
	if duplicateIDResp.Code != http.StatusConflict {
		t.Fatalf("expected explicit duplicate id to return 409, got %d: %s", duplicateIDResp.Code, duplicateIDResp.Body.String())
	}
}

// seedTeamAndRoom creates a team with workers and ensures a default room exists.
func seedTeamAndRoom(t *testing.T, h *OpenClawHandler, team TeamEntry, workers []ContainerEntry) ClawRoomEntry {
	t.Helper()
	if err := h.saveTeams([]TeamEntry{team}); err != nil {
		t.Fatalf("failed to seed team: %v", err)
	}
	if err := h.saveRegistry(workers); err != nil {
		t.Fatalf("failed to seed workers: %v", err)
	}
	h.mu.Lock()
	room, err := h.ensureDefaultRoomLocked(team)
	h.mu.Unlock()
	if err != nil {
		t.Fatalf("failed to ensure room: %v", err)
	}
	return room
}

func newTestTeam(id, name, leaderID string) TeamEntry {
	now := time.Now().UTC().Format(time.RFC3339)
	return TeamEntry{
		ID:        id,
		Name:      name,
		LeaderID:  leaderID,
		CreatedAt: now,
		UpdatedAt: now,
	}
}

func newTestWorker(name string, port int, tempDir, teamID, roleKind string) ContainerEntry {
	return ContainerEntry{
		Name:     name,
		Port:     port,
		Image:    "ghcr.io/openclaw/openclaw:latest",
		Token:    name + "-token",
		DataDir:  tempDir,
		TeamID:   teamID,
		RoleKind: roleKind,
	}
}

// encodeOpenAIResponse writes a minimal OpenAI chat completion response.
func encodeOpenAIResponse(w http.ResponseWriter, content string) {
	_ = json.NewEncoder(w).Encode(map[string]any{
		"choices": []map[string]any{{
			"message": map[string]any{"content": content},
		}},
	})
}
