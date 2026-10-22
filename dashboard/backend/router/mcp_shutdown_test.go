package router

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
)

func TestDashboardCloseDisconnectsMCPClients(t *testing.T) {
	dashboard := newMCPShutdownDashboard(t)
	serverID, sessionClosed := connectHTTPMCPForShutdown(t, dashboard)
	state, err := dashboard.mcpManager.GetServerStatus(serverID)
	if err != nil {
		t.Fatalf("get MCP server status before close: %v", err)
	}
	if state.Status != mcp.StatusConnected {
		t.Fatalf("MCP status before close = %q, want %q", state.Status, mcp.StatusConnected)
	}

	if closeErr := dashboard.Close(); closeErr != nil {
		t.Fatalf("close dashboard: %v", closeErr)
	}
	state, err = dashboard.mcpManager.GetServerStatus(serverID)
	if err != nil {
		t.Fatalf("get MCP server status after close: %v", err)
	}
	if state.Status != mcp.StatusDisconnected {
		t.Fatalf("MCP status after close = %q, want %q", state.Status, mcp.StatusDisconnected)
	}
	select {
	case sessionID := <-sessionClosed:
		if sessionID == "" {
			t.Fatal("HTTP MCP close request had no session ID")
		}
	default:
		t.Fatal("dashboard close returned before closing the HTTP MCP session")
	}
}

func TestDashboardCloseTerminatesStdioMCPProcess(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("requires POSIX process signals")
	}
	for _, mode := range []string{"eof", "term", "kill"} {
		t.Run(mode, func(t *testing.T) {
			testDashboardStdioShutdown(t, mode)
		})
	}
}

func testDashboardStdioShutdown(t *testing.T, mode string) {
	t.Helper()
	dashboard := newMCPShutdownDashboard(t)

	config, exitMarker := stdioMCPTestConfig(t, mode)
	if err := dashboard.mcpManager.AddServer(config); err != nil {
		t.Fatalf("add stdio MCP server: %v", err)
	}

	if err := dashboard.mcpManager.Connect(context.Background(), config.ID); err != nil {
		t.Fatalf("connect stdio MCP server: %v", err)
	}
	process := stdioHelperProcess(t, exitMarker)
	// Manager.Connect cancels its initialization context on success. The
	// process must remain usable until the manager explicitly closes it.
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if _, err := dashboard.mcpManager.ExecuteTool(ctx, config.ID, "ping", nil); err != nil {
		t.Fatalf("stdio connection did not survive initialization: %v", err)
	}
	httpServerID, _ := connectHTTPMCPForShutdown(t, dashboard)
	if _, err := os.Stat(exitMarker); !os.IsNotExist(err) {
		t.Fatalf("stdio helper exited before dashboard close: stat error = %v", err)
	}

	closeDashboardWithStdioProcess(t, dashboard, process)
	if _, err := os.Stat(exitMarker + ".eof"); err != nil {
		t.Fatalf("stdio helper did not observe EOF: %v", err)
	}
	if mode != "eof" {
		if _, err := os.Stat(exitMarker + ".term"); err != nil {
			t.Fatalf("stdio helper did not receive SIGTERM before escalation: %v", err)
		}
	}
	if _, err := os.Stat(exitMarker); mode != "kill" && err != nil {
		t.Fatalf("stdio helper did not exit after dashboard close: %v", err)
	}
	if _, err := os.Stat(exitMarker); mode == "kill" && !os.IsNotExist(err) {
		t.Fatalf("expected forced termination, but the helper returned normally: %v", err)
	}

	state, err := dashboard.mcpManager.GetServerStatus(config.ID)
	if err != nil {
		t.Fatalf("get stdio MCP status after close: %v", err)
	}
	if state.Status != mcp.StatusDisconnected {
		t.Fatalf("stdio MCP status after close = %q, want %q", state.Status, mcp.StatusDisconnected)
	}
	httpState, err := dashboard.mcpManager.GetServerStatus(httpServerID)
	if err != nil || httpState.Status != mcp.StatusDisconnected {
		t.Fatalf("HTTP MCP was not cleaned up: state=%+v, error=%v", httpState, err)
	}
}

func TestDashboardCloseCancelsStdioInitialization(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("requires POSIX process signals")
	}
	dashboard := newMCPShutdownDashboard(t)
	config, exitMarker := stdioMCPTestConfig(t, "kill")
	config.Connection.Env["SEMANTIC_ROUTER_MCP_STDIO_BLOCK_INITIALIZE"] = "1"
	if err := dashboard.mcpManager.AddServer(config); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	connectDone := make(chan error, 1)
	go func() { connectDone <- dashboard.mcpManager.Connect(ctx, config.ID) }()
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(exitMarker + ".initializing"); err == nil {
			break
		}
		select {
		case err := <-connectDone:
			t.Fatalf("connection returned before initialization could be canceled: %v", err)
		default:
		}
		if time.Now().After(deadline) {
			t.Fatal("stdio helper did not start initialization")
		}
		time.Sleep(time.Millisecond)
	}
	process := stdioHelperProcess(t, exitMarker)
	closeDashboardWithStdioProcess(t, dashboard, process)
	select {
	case err := <-connectDone:
		if err == nil {
			t.Fatal("expected interrupted initialization to fail")
		}
	case <-time.After(time.Second):
		t.Fatal("dashboard close returned without joining initialization")
	}
}

func closeDashboardWithStdioProcess(t *testing.T, dashboard *Server, process *os.Process) {
	t.Helper()
	closeDone := make(chan error, 1)
	go func() { closeDone <- dashboard.Close() }()
	select {
	case err := <-closeDone:
		if err != nil {
			t.Fatalf("close dashboard: %v", err)
		}
	case <-time.After(6 * time.Second):
		// Unblock cleanup even against the unfixed implementation.
		_ = process.Kill()
		<-closeDone
		t.Fatal("dashboard shutdown blocked on the stdio child")
	}
	if err := process.Signal(syscall.Signal(0)); !errors.Is(err, os.ErrProcessDone) && !errors.Is(err, syscall.ESRCH) {
		t.Fatalf("stdio helper still exists after shutdown: %v", err)
	}
	if err := dashboard.mcpManager.AddServer(&mcp.ServerConfig{ID: "after-close"}); err == nil || !strings.Contains(err.Error(), "database is closed") {
		t.Fatalf("workflow store was not closed after MCP cleanup: %v", err)
	}
}

func stdioMCPTestConfig(t *testing.T, mode string) (*mcp.ServerConfig, string) {
	t.Helper()
	exitMarker := filepath.Join(t.TempDir(), "stdio-helper-exited")
	config := &mcp.ServerConfig{
		ID:        "stdio-shutdown-test",
		Name:      "stdio shutdown test",
		Transport: mcp.TransportStdio,
		Connection: mcp.ConnectionConfig{
			Command: os.Args[0],
			Args:    []string{"-test.run=TestMCPStdioHelper"},
			Env: map[string]string{
				"SEMANTIC_ROUTER_MCP_STDIO_HELPER":      "1",
				"SEMANTIC_ROUTER_MCP_STDIO_EXIT_MARKER": exitMarker,
				"SEMANTIC_ROUTER_MCP_STDIO_EXIT_MODE":   mode,
				"GORACE":                                "atexit_sleep_ms=0",
			},
		},
	}
	if mode != "eof" {
		config.Connection.Cwd = filepath.Dir(exitMarker)
	}
	return config, exitMarker
}

func stdioHelperProcess(t *testing.T, exitMarker string) *os.Process {
	t.Helper()
	data, err := os.ReadFile(exitMarker + ".pid")
	if err != nil {
		t.Fatal(err)
	}
	pid, err := strconv.Atoi(string(data))
	if err != nil {
		t.Fatal(err)
	}
	process, err := os.FindProcess(pid)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = process.Kill()
		_ = process.Release()
	})
	return process
}

func TestMCPStdioHelper(t *testing.T) {
	if os.Getenv("SEMANTIC_ROUTER_MCP_STDIO_HELPER") != "1" {
		return
	}
	exitMarker := os.Getenv("SEMANTIC_ROUTER_MCP_STDIO_EXIT_MARKER")
	mode := os.Getenv("SEMANTIC_ROUTER_MCP_STDIO_EXIT_MODE")
	termination := make(chan os.Signal, 1)
	if mode == "term" || mode == "kill" {
		signal.Notify(termination, syscall.SIGTERM)
		defer signal.Stop(termination)
	}
	if err := os.WriteFile(exitMarker+".pid", []byte(strconv.Itoa(os.Getpid())), 0o644); err != nil {
		t.Fatal(err)
	}

	defer func() {
		_ = os.WriteFile(exitMarker, nil, 0o644)
	}()

	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		var request struct {
			JSONRPC string          `json:"jsonrpc"`
			ID      json.RawMessage `json:"id"`
			Method  string          `json:"method"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &request); err != nil || len(request.ID) == 0 {
			continue
		}

		var result interface{}
		switch request.Method {
		case "initialize":
			_ = os.WriteFile(exitMarker+".initializing", nil, 0o644)
			if os.Getenv("SEMANTIC_ROUTER_MCP_STDIO_BLOCK_INITIALIZE") == "1" {
				continue
			}
			result = map[string]interface{}{
				"protocolVersion": "2025-06-18",
				"serverInfo": map[string]string{
					"name":    "stdio-shutdown-test",
					"version": "1.0.0",
				},
				"capabilities": map[string]interface{}{},
			}
		case "tools/list":
			result = map[string]interface{}{"tools": []interface{}{}}
		case "tools/call":
			result = map[string]interface{}{"content": []interface{}{}}
		default:
			result = map[string]interface{}{}
		}

		response, err := json.Marshal(map[string]interface{}{
			"jsonrpc": "2.0",
			"id":      request.ID,
			"result":  result,
		})
		if err != nil {
			continue
		}
		_, _ = fmt.Fprintf(os.Stdout, "%s\n", response)
	}
	_ = os.WriteFile(exitMarker+".eof", nil, 0o644)
	if mode == "term" || mode == "kill" {
		for range termination {
			_ = os.WriteFile(exitMarker+".term", nil, 0o644)
			if mode == "term" {
				return
			}
		}
	}
}
