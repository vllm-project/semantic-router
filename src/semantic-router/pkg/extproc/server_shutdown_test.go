package extproc

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"syscall"
	"testing"
	"time"
)

const (
	sigtermDrainChildEnv    = "SR_SIGTERM_DRAIN_CHILD"
	sigtermDrainChildPort   = "SR_SIGTERM_DRAIN_CHILD_PORT"
	sigtermDrainReadyMarker = "SIGTERM_DRAIN_CHILD_READY"
	sigtermDrainMarker      = "SIGTERM_DRAIN_COMPLETE"
	sigtermDrainSlowHookDur = 1500 * time.Millisecond
)

// TestServerStopRunsShutdownHooksAfterGracefulDrainBeforeProcessExits spawns
// a child process (self-exec of the test binary) that runs a real Server
// with a slow shutdown hook registered, sends SIGTERM, and asserts the
// process only exits once that hook has completed.
//
// This is issue #2470 finding #6: cmd/runtime_bootstrap.go used to register
// its own independent signal.Notify + unconditional os.Exit(0), racing
// Server's own signal.Notify + GracefulStop() — the hard exit had no reason
// to lose that race, so it could kill the process mid-drain. Server is now
// the sole SIGINT/SIGTERM owner (see Start): no other signal.Notify for
// these signals exists in the shutdown path, and nothing calls os.Exit —
// the process only terminates once Start() returns, which only happens
// after Stop()'s graceful drain and every registered shutdown hook finish.
func TestServerStopRunsShutdownHooksAfterGracefulDrainBeforeProcessExits(t *testing.T) {
	if os.Getenv(sigtermDrainChildEnv) == "1" {
		runSigtermDrainChild()
		return
	}

	if runtime.GOOS == "windows" {
		t.Skip("POSIX signal test")
	}

	port, err := freeTCPPort()
	if err != nil {
		t.Fatalf("failed to allocate a free port: %v", err)
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestServerStopRunsShutdownHooksAfterGracefulDrainBeforeProcessExits$")
	cmd.Env = append(os.Environ(),
		sigtermDrainChildEnv+"=1",
		sigtermDrainChildPort+"="+strconv.Itoa(port),
	)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("StdoutPipe() error = %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("Start() error = %v", err)
	}

	// Drain stdout continuously for the life of the test instead of reading
	// after cmd.Wait() returns: Wait documents that it closes the pipe once
	// the child exits, discarding any output not already read by then, so a
	// post-Wait ReadAll can silently lose the drain marker.
	lines := make(chan string, 16)
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			lines <- scanner.Text()
		}
		close(lines)
	}()

	readyLine, err := readLineWithTimeout(lines, 10*time.Second)
	if err != nil {
		_ = cmd.Process.Kill()
		t.Fatalf("waiting for child readiness marker: %v", err)
	}
	if readyLine != sigtermDrainReadyMarker {
		_ = cmd.Process.Kill()
		t.Fatalf("unexpected child readiness line = %q", readyLine)
	}

	sigSentAt := time.Now()
	if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatalf("Signal(SIGTERM) error = %v", err)
	}

	drainLine, err := readLineWithTimeout(lines, sigtermDrainSlowHookDur+10*time.Second)
	elapsed := time.Since(sigSentAt)
	sawDrainMarker := err == nil && drainLine == sigtermDrainMarker
	if !sawDrainMarker || elapsed < sigtermDrainSlowHookDur {
		_ = cmd.Process.Kill()
		t.Fatalf(
			"child did not complete its shutdown hook %s after SIGTERM (drain marker seen = %v, read error = %v); "+
				"a SIGTERM must not terminate the process before Server.Stop's registered shutdown hooks finish",
			elapsed, sawDrainMarker, err,
		)
	}

	if err := cmd.Wait(); err != nil {
		t.Fatalf("child process exited with error after completing its shutdown hook: %v", err)
	}
}

func freeTCPPort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer func() { _ = l.Close() }()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func readLineWithTimeout(lines <-chan string, timeout time.Duration) (string, error) {
	select {
	case line, ok := <-lines:
		if !ok {
			return "", io.EOF
		}
		return line, nil
	case <-time.After(timeout):
		return "", os.ErrDeadlineExceeded
	}
}

// runSigtermDrainChild runs only inside the self-exec'd child process. It
// starts a real Server on the port the parent chose, with a slow shutdown
// hook registered, then blocks in Start(). The process should only end once
// Start() returns — which only happens after SIGTERM triggers Stop()'s
// graceful drain and hook.
func runSigtermDrainChild() {
	port, err := strconv.Atoi(os.Getenv(sigtermDrainChildPort))
	if err != nil {
		_, _ = os.Stderr.WriteString("invalid " + sigtermDrainChildPort + ": " + err.Error() + "\n")
		os.Exit(2)
	}

	server := &Server{
		service: NewRouterService(&OpenAIRouter{}),
		port:    port,
	}
	server.RegisterShutdownHook(func() {
		time.Sleep(sigtermDrainSlowHookDur)
		_, _ = os.Stdout.WriteString(sigtermDrainMarker + "\n")
	})

	go func() {
		waitForPortListening(port, 5*time.Second)
		_, _ = os.Stdout.WriteString(sigtermDrainReadyMarker + "\n")
	}()

	_ = server.Start()
}

// waitForPortListening polls a real, observable condition (the port
// accepting connections) instead of sleeping a guessed duration, so the
// readiness signal to the parent is never sent before Server.Start has
// actually registered its signal handler and started serving.
func waitForPortListening(port int, timeout time.Duration) {
	deadline := time.Now().Add(timeout)
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 100*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
}
