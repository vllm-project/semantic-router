package modelresearch

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

type commandSpec struct {
	Dir  string
	Name string
	Args []string
	Env  []string
}

type commandRunner func(ctx context.Context, spec commandSpec, onLine func(stream, line string)) error

func defaultCommandRunner(ctx context.Context, spec commandSpec, onLine func(stream, line string)) error {
	if err := validateCommandSpec(spec); err != nil {
		return err
	}
	// #nosec G204 -- executable is restricted to validated Python interpreters and args are assembled from internal recipes.
	cmd := exec.CommandContext(ctx, spec.Name, spec.Args...)
	cmd.Dir = spec.Dir
	if len(spec.Env) > 0 {
		cmd.Env = append(cmd.Environ(), spec.Env...)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}

	if err := cmd.Start(); err != nil {
		return err
	}

	var wg sync.WaitGroup
	streamOutput := func(stream string, reader io.Reader) {
		defer wg.Done()
		scanner := bufio.NewScanner(reader)
		buf := make([]byte, 0, 64*1024)
		scanner.Buffer(buf, 1024*1024)
		for scanner.Scan() {
			onLine(stream, scanner.Text())
		}
	}

	wg.Add(2)
	go streamOutput("stdout", stdout)
	go streamOutput("stderr", stderr)
	wg.Wait()

	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("%s %v failed: %w", spec.Name, spec.Args, err)
	}
	return nil
}

func validateCommandSpec(spec commandSpec) error {
	if strings.TrimSpace(spec.Name) == "" {
		return fmt.Errorf("command executable is required")
	}
	if strings.ContainsAny(spec.Name, "\r\n") {
		return fmt.Errorf("command executable contains invalid control characters")
	}

	switch filepath.Base(spec.Name) {
	case "python", "python3":
		return nil
	default:
		return fmt.Errorf("unsupported executable %q", spec.Name)
	}
}
