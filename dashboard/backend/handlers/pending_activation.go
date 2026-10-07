package handlers

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Pending activations. The Dashboard holds no container runtime: when a saved
// config needs the Router (and Envoy) created anew -- first-run setup, or a
// change the running containers can't take -- the `vllm-sr serve` that owns
// the stack applies it. Two files beside the runtime config carry the hand-off
// (the CLI's half is src/vllm-sr/cli/pending_activation.py): the heartbeat of
// an attached CLI, and the Dashboard's record of the saved activation.
const (
	serveHeartbeatSuffix    = ".serve-heartbeat.json"
	pendingActivationSuffix = ".pending-activation.json"
	// The CLI beats every two seconds; a few missed beats mean it stopped.
	serveHeartbeatFreshness = 10 * time.Second
	maxServeHeartbeatBytes  = 4 << 10
)

// The heartbeat's state while the attached CLI waits to apply an activation;
// otherwise it starts the stack. A heartbeat without a state comes from a CLI
// that beat only while it waited for setup.
const serveWaiting = "waiting"

type activationReason string

const (
	activationReasonSetup   activationReason = "setup"
	activationReasonRestart activationReason = "restart"
)

type pendingActivationRecord struct {
	Reason       activationReason `json:"reason"`
	RecordedAt   string           `json:"recorded_at"`
	ConfigSHA256 string           `json:"config_sha256"`
	Detail       string           `json:"detail,omitempty"`
}

func pendingActivationPath(configPath string, suffix string) string {
	return strings.TrimSuffix(configPath, filepath.Ext(configPath)) + suffix
}

// serveState returns what an attached `vllm-sr serve` does, or "" when none
// beats.
func serveState(configPath string) string {
	path := pendingActivationPath(configPath, serveHeartbeatSuffix)
	info, err := os.Stat(path)
	if err != nil || time.Since(info.ModTime()) >= serveHeartbeatFreshness || info.Size() > maxServeHeartbeatBytes {
		return ""
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	var heartbeat struct {
		State string `json:"state"`
	}
	if json.Unmarshal(data, &heartbeat) != nil || heartbeat.State == "" {
		return serveWaiting
	}
	return heartbeat.State
}

// serveAttached reports whether a `vllm-sr serve` waits to apply this config.
func serveAttached(configPath string) bool {
	return serveState(configPath) == serveWaiting
}

// serveRunning reports whether a `vllm-sr serve` is starting the stack or
// waits to apply an activation.
func serveRunning(configPath string) bool {
	return serveState(configPath) != ""
}

func pendingActivationRecorded(configPath string) bool {
	_, err := os.Lstat(pendingActivationPath(configPath, pendingActivationSuffix))
	return err == nil
}

// recordPendingActivation tells the attached CLI, or the next `vllm-sr serve`,
// that config (already written to configPath) waits for the containers to be
// created anew.
func recordPendingActivation(configPath string, config []byte, reason activationReason, detail string) error {
	digest := sha256.Sum256(config)
	data, err := json.Marshal(pendingActivationRecord{
		Reason:       reason,
		RecordedAt:   time.Now().UTC().Format(time.RFC3339),
		ConfigSHA256: hex.EncodeToString(digest[:]),
		Detail:       detail,
	})
	if err != nil {
		return err
	}
	target := pendingActivationPath(configPath, pendingActivationSuffix)
	staged, err := os.CreateTemp(filepath.Dir(target), ".pending-activation-*")
	if err != nil {
		return err
	}
	defer os.Remove(staged.Name())
	if _, err := staged.Write(data); err != nil {
		staged.Close()
		return err
	}
	if err := staged.Chmod(0o644); err != nil {
		staged.Close()
		return err
	}
	if err := staged.Close(); err != nil {
		return err
	}
	if err := os.Rename(staged.Name(), target); err != nil {
		return fmt.Errorf("record the pending activation: %w", err)
	}
	return nil
}

// setupActivatedMessage says who starts the Router now that setup is done.
func setupActivatedMessage(configPath string) string {
	services := "The Router is"
	if managedStackRunsEnvoy() {
		services = "The Router and Envoy are"
	}
	if serveAttached(configPath) {
		return fmt.Sprintf("Setup saved. %s starting.", services)
	}
	return "Setup saved. Run `vllm-sr serve` to start the Router."
}

// restartRequiredMessage says who applies a saved change that needs the
// containers created anew.
func restartRequiredMessage(configPath string) string {
	if serveAttached(configPath) {
		return "Restart required: `vllm-sr serve` is applying it."
	}
	return "Restart required: run `vllm-sr serve` to apply."
}
