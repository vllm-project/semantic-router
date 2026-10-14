package router

import (
	"context"
	"os"
	"runtime"
	"testing"
	"time"
)

func TestDashboardCloseCancelsMCPConnectionTest(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("requires POSIX process signals")
	}
	dashboard := newMCPShutdownDashboard(t)
	config, exitMarker := stdioMCPTestConfig(t, "kill")
	config.Connection.Env["SEMANTIC_ROUTER_MCP_STDIO_BLOCK_INITIALIZE"] = "1"
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	testDone := make(chan error, 1)
	testFinished := make(chan struct{})
	go func() {
		defer close(testFinished)
		testDone <- dashboard.mcpManager.TestConnection(ctx, config)
	}()
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(exitMarker + ".initializing"); err == nil {
			break
		}
		select {
		case err := <-testDone:
			t.Fatalf("connection test returned before initialization: %v", err)
		default:
		}
		if time.Now().After(deadline) {
			t.Fatal("stdio helper did not start initialization")
		}
		time.Sleep(time.Millisecond)
	}
	process := stdioHelperProcess(t, exitMarker)
	t.Cleanup(func() {
		cancel()
		select {
		case <-testFinished:
		case <-time.After(5 * time.Second):
			_ = process.Kill()
			t.Error("connection test did not finish cleanup")
		}
	})
	closeDashboardWithStdioProcess(t, dashboard, process)
	select {
	case err := <-testDone:
		if err == nil {
			t.Fatal("expected interrupted connection test to fail")
		}
	case <-time.After(time.Second):
		t.Fatal("dashboard close returned without joining the connection test")
	}
}

func TestDashboardCloseRejectsMCPConnectionTest(t *testing.T) {
	dashboard := newMCPShutdownDashboard(t)
	config, exitMarker := stdioMCPTestConfig(t, "eof")
	if err := dashboard.Close(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := dashboard.mcpManager.TestConnection(ctx, config); err == nil {
		t.Fatal("closed manager accepted a connection test")
	}
	if _, err := os.Stat(exitMarker + ".pid"); !os.IsNotExist(err) {
		t.Fatal("connection test spawned a child after dashboard close")
	}
}
