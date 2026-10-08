package modelservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

// refuseGPUOnlyOnCPU fails an implicit deployment of a built-in model that runs
// on a GPU only when it would land on the CPU: on cpu, or on auto on a host
// where the runtime finds no GPU.
func refuseGPUOnlyOnCPU(deployments map[string]config.ModelDeployment, auto string) error {
	names := make([]string, 0, len(deployments))
	for name := range deployments {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		deployment := deployments[name].WithDefaults()
		if !deployment.Managed() || !config.ImplicitDeploymentRequiresGPU(name, deployment) {
			continue
		}
		onCPU := false
		for _, placement := range deployment.Placements() {
			if placement.Endpoint == "" && (placement.Device == "cpu" || placement.Device == autoDevice && auto == "cpu") {
				onCPU = true
			}
		}
		if onCPU {
			return fmt.Errorf("model_runtime deployment %q: %s runs on a GPU only, and the model runtime finds no GPU on this host; "+
				"serve the Router on a GPU host (vllm-sr serve --platform amd or nvidia), or choose Vela-2.0-0.3B or Vela-2.0-0.8B as the decision model",
				name, deployment.Artifact)
		}
	}
	return nil
}
