package mcp

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
)

const (
	stdioEOFGracePeriod       = time.Second
	stdioTerminateGracePeriod = time.Second
)

// ownedStdioClient bounds shutdown of the process started by the SDK, including
// reaping it when the SDK returns early after a pipe-close error.
type ownedStdioClient struct {
	client.MCPClient
	cmd       *exec.Cmd
	cancel    context.CancelFunc
	closeOnce sync.Once
	closeErr  error
}

func (c *Client) createStdioClient(ctx context.Context) (client.MCPClient, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	log.Printf("[MCP-Client] Creating Stdio client: argument_count=%d", len(c.config.Connection.Args))

	env := os.Environ()
	for k, v := range c.config.Connection.Env {
		env = append(env, fmt.Sprintf("%s=%s", k, v))
	}

	// Connect's context is canceled after initialization. The child must instead
	// live until Close, including cleanup after an initialization failure.
	processCtx, cancel := context.WithCancel(context.Background())
	var cmd *exec.Cmd
	commandOption := transport.WithCommandFunc(func(_ context.Context, command string, env, args []string) (*exec.Cmd, error) {
		cmd = exec.CommandContext(processCtx, command, args...)
		cmd.Env = env
		cmd.Dir = c.config.Connection.Cwd
		cmd.Cancel = func() error {
			err := cmd.Process.Signal(syscall.SIGTERM)
			if err != nil && !errors.Is(err, os.ErrProcessDone) {
				return cmd.Process.Kill()
			}
			return err
		}
		// os/exec escalates to Kill if SIGTERM does not make the child exit.
		cmd.WaitDelay = stdioTerminateGracePeriod
		return cmd, nil
	})
	mcpClient, err := client.NewStdioMCPClientWithOptions(
		c.config.Connection.Command, env, c.config.Connection.Args, commandOption,
	)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create stdio client: %w", err)
	}
	return &ownedStdioClient{MCPClient: mcpClient, cmd: cmd, cancel: cancel}, nil
}

func (c *ownedStdioClient) Close() error {
	// Initialization failure and concurrent Disconnect can both close the same
	// SDK client. Serialize them and make every caller wait for process cleanup.
	c.closeOnce.Do(func() {
		defer c.cancel()
		done := make(chan error, 1)
		go func() {
			err := c.MCPClient.Close()
			// SDK Close can fail on a pipe before reaching Wait. Check only after
			// it returns so the SDK and this fallback never wait concurrently.
			if c.cmd != nil && c.cmd.ProcessState == nil {
				err = errors.Join(err, c.cmd.Wait())
			}
			done <- err
		}()

		// SDK Close closes stdin before waiting. Give a cooperative child time to
		// exit on EOF, then cancel its process context to terminate and escalate.
		timer := time.NewTimer(stdioEOFGracePeriod)
		defer timer.Stop()
		select {
		case c.closeErr = <-done:
		case <-timer.C:
			c.cancel()
			c.closeErr = <-done
		}
	})
	return c.closeErr
}
