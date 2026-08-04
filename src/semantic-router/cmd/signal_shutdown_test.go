package main

import (
	"context"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

const (
	signalChildEnv     = "SR_SIGNAL_SHUTDOWN_CHILD"
	signalChildServing = "serving"
	signalChildStartup = "startup"

	markerServerDrained = "marker:server-drained"
	markerStoreReleased = "marker:vector-store-released"
	markerTracingFlush  = "marker:tracing-flushed"

	// childDrainDuration makes the "server drain" hook slow enough that any
	// cleanup racing it instead of waiting for it shows up as reordered
	// markers rather than a flake.
	childDrainDuration = 300 * time.Millisecond
	// childCleanupDuration makes the resource-release hook slow enough that a
	// main goroutine returning without waiting for the handler truncates it,
	// rather than getting away with it by a few microseconds.
	childCleanupDuration = 300 * time.Millisecond
)

func TestShutdownRegistryRunsHooksInReverseRegistrationOrder(t *testing.T) {
	registry := newShutdownRegistry()
	order := make([]string, 0, 3)

	registry.register(func() { order = append(order, "vector-store") })
	registry.register(func() { order = append(order, "middle") })
	registry.register(func() { order = append(order, "server-stop") })
	registry.register(nil)

	registry.runInReverse()

	want := []string{"server-stop", "middle", "vector-store"}
	if !reflect.DeepEqual(order, want) {
		t.Fatalf("hook order = %v, want %v — hooks must tear down in reverse registration order so a drain never outlives the resources it uses", order, want)
	}
}

// TestShutdownRegistryConcurrentRegisterAndRun exercises the sharing pattern
// the signal handler creates: main keeps appending hooks while startup runs,
// and the signal goroutine may run them at any moment. Run with -race.
func TestShutdownRegistryConcurrentRegisterAndRun(t *testing.T) {
	registry := newShutdownRegistry()

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			registry.register(func() {})
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			registry.runInReverse()
		}
	}()
	wg.Wait()
}

// TestSignalHandlerDrainsBeforeReleasingSharedResources asserts the full
// process-level shutdown order under a real SIGTERM: the last-registered hook
// (server.Stop in main) runs first and completes before earlier hooks release
// the resources in-flight requests are still using, tracing is flushed after
// all of them, and the process exits cleanly instead of being killed by the
// default signal disposition.
//
// The child also mimics main returning the moment server.Stop lets Start
// return, so the missing markers catch a main that exits out from under the
// still-running handler.
func TestSignalHandlerDrainsBeforeReleasingSharedResources(t *testing.T) {
	lines := runSignalShutdownChild(t, signalChildServing)

	want := []string{markerServerDrained, markerStoreReleased, markerTracingFlush}
	if !reflect.DeepEqual(lines, want) {
		t.Fatalf("shutdown markers = %v, want %v", lines, want)
	}
}

// TestSignalHandlerRunsHooksWhenSignalArrivesBeforeServerExists covers the
// window this whole ownership split exists for: model download, runtime init
// and warmup take minutes, so a SIGTERM routinely lands before any server has
// been constructed. Cleanup registered so far must still run, tracing must
// still flush, and the process must still exit 0 — none of which happens if
// the only handler lives inside the server.
func TestSignalHandlerRunsHooksWhenSignalArrivesBeforeServerExists(t *testing.T) {
	lines := runSignalShutdownChild(t, signalChildStartup)

	want := []string{markerStoreReleased, markerTracingFlush}
	if !reflect.DeepEqual(lines, want) {
		t.Fatalf("shutdown markers = %v, want %v", lines, want)
	}
}

// runSignalShutdownChild re-executes this test binary in child mode, where it
// installs the real signal handler and sends itself a real SIGTERM. The child
// is a separate process because the handler ends in os.Exit: only a child can
// prove the process exits, and exits cleanly, after the hooks complete.
func runSignalShutdownChild(t *testing.T, mode string) []string {
	t.Helper()

	if runtime.GOOS == "windows" {
		t.Skip("POSIX signal test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// #nosec G204 -- os.Args[0] is this test binary, re-executed with a fixed
	// -test.run filter; neither argument carries external input.
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestSignalHandlerChildProcess$")
	cmd.Env = append(os.Environ(), signalChildEnv+"="+mode)

	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("child process (%s mode) exited with error = %v; a SIGTERM must be handled, not kill the process\nstdout:\n%s", mode, err, out)
	}

	lines := make([]string, 0, 3)
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, "marker:") {
			lines = append(lines, line)
		}
	}
	return lines
}

// TestSignalHandlerChildProcess runs only inside the re-executed child. It
// registers hooks in the same order main does, installs the real signal
// handler, and raises a real SIGTERM at itself; the handler's os.Exit is what
// ends the process.
func TestSignalHandlerChildProcess(t *testing.T) {
	mode := os.Getenv(signalChildEnv)
	if mode == "" {
		t.Skip("child-process-only test")
	}

	shutdownTracingHook = func() { emitShutdownMarker(markerTracingFlush) }

	registry := newShutdownRegistry()
	// Registered first, exactly like the vector store hook in
	// initializeRuntimeDependencies, so reverse order runs it last.
	registry.register(func() {
		time.Sleep(childCleanupDuration)
		emitShutdownMarker(markerStoreReleased)
	})
	registerSignalHandler(registry)

	drained := make(chan struct{})
	if mode == signalChildServing {
		// Stands in for server.Stop, which main registers last and which
		// blocks until in-flight requests have drained. Closing drained is
		// Start returning in main.
		registry.register(func() {
			time.Sleep(childDrainDuration)
			emitShutdownMarker(markerServerDrained)
			close(drained)
		})
	}

	if err := syscall.Kill(os.Getpid(), syscall.SIGTERM); err != nil {
		t.Fatalf("Kill(SIGTERM) error = %v", err)
	}

	if mode == signalChildServing {
		select {
		case <-drained:
		case <-time.After(20 * time.Second):
			t.Fatal("SIGTERM did not reach the registered handler")
		}
		// Exactly what main does after startExtProcServerOrFatal returns.
		registry.awaitShutdownExit()
		os.Exit(0)
	}

	// Startup mode: main is still deep in initialization, so the handler's
	// own os.Exit is what ends the process.
	time.Sleep(20 * time.Second)
	t.Fatal("SIGTERM did not reach the registered handler")
}

func emitShutdownMarker(marker string) {
	_, _ = os.Stdout.WriteString(marker + "\n")
}
