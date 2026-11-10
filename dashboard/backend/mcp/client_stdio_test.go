package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"syscall"
	"testing"
	"time"

	sdkclient "github.com/mark3labs/mcp-go/client"
)

func TestStdioCloseReapsProcess(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("requires POSIX process signals")
	}
	for _, mode := range []string{"eof", "kill", "stderr-error"} {
		t.Run(mode, func(t *testing.T) {
			instance, process := newStdioCloseTestClient(t, mode)
			owned := instance.mcpClient.(*ownedStdioClient)
			if mode == "stderr-error" {
				stderr, ok := sdkclient.GetStderr(owned.MCPClient.(*sdkclient.Client))
				if !ok {
					t.Fatal("expected stdio stderr pipe")
				}
				// Close the parent's reader to make SDK Close fail before cmd.Wait.
				if err := stderr.(io.Closer).Close(); err != nil {
					t.Fatal(err)
				}
			}

			const callers = 8
			type closeResult struct {
				err        error
				processErr error
			}
			start := make(chan struct{})
			results := make(chan closeResult, callers)
			for range callers {
				go func() {
					<-start
					err := owned.Close()
					results <- closeResult{err: err, processErr: process.Signal(syscall.Signal(0))}
				}()
			}
			close(start)
			deadline := time.NewTimer(5 * time.Second)
			defer deadline.Stop()
			for range callers {
				select {
				case result := <-results:
					if mode == "eof" && result.err != nil {
						t.Errorf("graceful close: %v", result.err)
					}
					if mode == "stderr-error" && !errors.Is(result.err, os.ErrClosed) {
						t.Errorf("Close lost the pipe error: %v", result.err)
					}
					var exitErr *exec.ExitError
					if mode != "eof" && !errors.As(result.err, &exitErr) {
						t.Errorf("Close did not report the terminated child: %v", result.err)
					}
					if !errors.Is(result.processErr, os.ErrProcessDone) && !errors.Is(result.processErr, syscall.ESRCH) {
						t.Errorf("Close returned before the child was reaped: %v", result.processErr)
					}
				case <-deadline.C:
					t.Fatal("Close blocked on the stdio child")
				}
			}
		})
	}
}

func newStdioCloseTestClient(t *testing.T, mode string) (*Client, *os.Process) {
	t.Helper()
	marker := filepath.Join(t.TempDir(), "stdio-helper.pid")
	instance, err := NewClient(&ServerConfig{
		ID: "stdio-close-test", Transport: TransportStdio,
		Connection: ConnectionConfig{
			Command: os.Args[0],
			Args:    []string{"-test.run=^TestStdioCloseHelper$"},
			Env: map[string]string{
				"SEMANTIC_ROUTER_STDIO_CLOSE_HELPER": marker,
				"SEMANTIC_ROUTER_STDIO_CLOSE_MODE":   mode,
				"GORACE":                             "atexit_sleep_ms=0",
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = instance.Disconnect() })
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if connectErr := instance.Connect(ctx); connectErr != nil {
		t.Fatal(connectErr)
	}
	data, err := os.ReadFile(marker)
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
		// Also clean up when run against the implementation that skips Wait.
		_ = process.Kill()
		_ = instance.Disconnect()
		if err := process.Signal(syscall.Signal(0)); err == nil {
			_, _ = process.Wait()
		}
		_ = process.Release()
	})
	return instance, process
}

func TestStdioCloseHelper(t *testing.T) {
	marker := os.Getenv("SEMANTIC_ROUTER_STDIO_CLOSE_HELPER")
	if marker == "" {
		return
	}
	signal.Ignore(syscall.SIGTERM)
	if err := os.WriteFile(marker, []byte(strconv.Itoa(os.Getpid())), 0o644); err != nil {
		t.Fatal(err)
	}
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		var request struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &request); err != nil || len(request.ID) == 0 {
			continue
		}
		var result interface{}
		switch request.Method {
		case "initialize":
			result = map[string]interface{}{
				"protocolVersion": "2025-06-18",
				"serverInfo":      map[string]string{"name": "stdio-close-test", "version": "1.0.0"},
				"capabilities":    map[string]interface{}{},
			}
		case "tools/list":
			result = map[string]interface{}{"tools": []interface{}{}}
		default:
			result = map[string]interface{}{}
		}
		response, err := json.Marshal(map[string]interface{}{"jsonrpc": "2.0", "id": request.ID, "result": result})
		if err != nil {
			t.Fatal(err)
		}
		_, _ = fmt.Fprintf(os.Stdout, "%s\n", response)
	}
	if os.Getenv("SEMANTIC_ROUTER_STDIO_CLOSE_MODE") != "eof" {
		for {
			time.Sleep(time.Hour)
		}
	}
}
