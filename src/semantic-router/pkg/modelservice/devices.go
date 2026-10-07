package modelservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

// autoDevice is the device value that lets the runtime place a model.
const autoDevice = "auto"

// autoDeviceTimeout bounds the runtime's devices command, which imports
// PyTorch and enumerates the GPUs.
const autoDeviceTimeout = 2 * time.Minute

// deviceLabel is the shape of a device the runtime reports: an accelerator
// name with an optional :N.
var deviceLabel = regexp.MustCompile(`^[a-z][a-z0-9_]*(:[0-9]+)?$`)

// queryAutoDevice asks the runtime command which device --device auto takes
// on this host (`<command> devices`), so deployments on auto are planned by
// the runtime's own choice: its accelerator plugins and their auto_priority.
func queryAutoDevice(command []string) (string, error) {
	if len(command) == 0 {
		return "", errors.New("no runtime command")
	}
	ctx, cancel := context.WithTimeout(context.Background(), autoDeviceTimeout)
	defer cancel()
	args := append(append([]string(nil), command[1:]...), "devices")
	cmd := exec.CommandContext(ctx, command[0], args...) //nolint:gosec // the command comes from Router configuration and environment
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	cmd.WaitDelay = time.Second
	output, err := cmd.Output()
	if err != nil {
		if message := lastLine(stderr.String()); message != "" {
			return "", fmt.Errorf("%w: %s", err, message)
		}
		return "", err
	}
	var report struct {
		Auto string `json:"auto"`
	}
	if err := json.Unmarshal(output, &report); err != nil {
		return "", fmt.Errorf("unexpected devices output: %w", err)
	}
	if report.Auto == autoDevice || !deviceLabel.MatchString(report.Auto) {
		return "", fmt.Errorf("unexpected auto device %q", report.Auto)
	}
	return report.Auto, nil
}

func lastLine(text string) string {
	lines := strings.Split(strings.TrimSpace(text), "\n")
	return strings.TrimSpace(lines[len(lines)-1])
}
