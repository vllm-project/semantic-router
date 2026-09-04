package evaluationplane

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"
	"unicode/utf8"
)

const workerModule = "cli.evaluation.worker"

const (
	maxWorkerEventLineBytes = 16 * 1024
	maxWorkerMessageBytes   = 512
	maxWorkerRecordCount    = 100_000_000
	workerProcessWaitDelay  = 5 * time.Second
)

const (
	routerEvaluationCredentialEnv = "VLLM_SR_EVALUATION_ROUTER_API_KEY" //nolint:gosec // Environment variable name, not a credential.
	routerManagementCredentialEnv = "VLLM_SR_DASHBOARD_RECIPE_TOKEN"    //nolint:gosec // Environment variable name, not a credential.
)

// CredentialProvider is the narrow recipe-store seam required by evaluation
// execution. Missing credentials must be reported as os.ErrNotExist.
type CredentialProvider interface {
	ManagementCredential() (string, error)
}

type ProcessSpec struct {
	ManifestPath string
	StorePath    string
}

// Process is injectable for tests, but its input deliberately contains no
// command, argument, environment, or executable override.
type Process interface {
	Run(context.Context, ProcessSpec, func(WorkerEvent) error) error
}

type CommandProcess struct {
	pythonPath     string
	envoyAPIKeyEnv string
}

func NewCommandProcess(pythonPath string) *CommandProcess {
	return &CommandProcess{pythonPath: strings.TrimSpace(pythonPath)}
}

func (p *CommandProcess) Run(ctx context.Context, spec ProcessSpec, emit func(WorkerEvent) error) error {
	if p.pythonPath == "" {
		return fmt.Errorf("python interpreter is not configured")
	}
	staging, err := prepareWorkerStaging(spec)
	if err != nil {
		return err
	}
	defer staging.cleanup()
	workerEnv, err := p.workerEnvironment(staging.manifestPath)
	if err != nil {
		return err
	}
	// The executable and argument grammar are server configured and fixed.
	//nolint:gosec
	cmd := exec.CommandContext(
		ctx,
		p.pythonPath,
		"-m", workerModule,
		"--manifest", staging.manifestPath,
		"--store", staging.storePath,
		"--events-stdout",
	)
	configureWorkerProcessGroup(cmd)
	cmd.Cancel = func() error { return terminateWorkerProcessGroup(cmd) }
	cmd.WaitDelay = workerProcessWaitDelay
	workerEnv = append(workerEnv, "PYTHONUNBUFFERED=1")
	cmd.Env = workerEnv
	cmd.Stderr = io.Discard // Worker stderr may contain private cases or provider diagnostics.
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("open evaluation worker stdout: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start evaluation worker: %w", err)
	}

	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 4*1024), maxWorkerEventLineBytes)
	for scanner.Scan() {
		event, decodeErr := decodeWorkerEvent(scanner.Bytes())
		if decodeErr != nil {
			_ = terminateWorkerProcessGroup(cmd)
			_ = cmd.Wait()
			return decodeErr
		}
		if err := emit(event); err != nil {
			_ = terminateWorkerProcessGroup(cmd)
			_ = cmd.Wait()
			return err
		}
	}
	scanErr := scanner.Err()
	if scanErr != nil {
		// A protocol line can exceed Scanner's bound while the worker and its
		// descendants keep running. Kill before Wait so malformed stdout cannot
		// retain a concurrency slot until the run TTL.
		_ = terminateWorkerProcessGroup(cmd)
		_ = cmd.Wait()
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("read evaluation worker events: %w", scanErr)
	}
	waitErr := cmd.Wait()
	// A successful worker may still have spawned a detached descendant that no
	// longer owns stdout. Reap the server-owned process group before importing
	// any evidence or releasing the run slot.
	_ = terminateWorkerProcessGroup(cmd)
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if waitErr != nil {
		return fmt.Errorf("evaluation worker exited unsuccessfully: %w", waitErr)
	}
	return staging.importEvidence()
}

func (p *CommandProcess) workerEnvironment(manifestPath string) ([]string, error) {
	manifest, _, err := readRunManifestStrict(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read staged evaluation manifest: %w", err)
	}
	environment := allowlistedWorkerEnvironment()
	var envoyCredential string
	if ref := manifest.Target.EnvoyAPIKey; ref != nil {
		if p.envoyAPIKeyEnv != "" && ref.Env != p.envoyAPIKeyEnv {
			return nil, fmt.Errorf("staged evaluation manifest has an unsupported Envoy credential reference")
		}
		value, present := os.LookupEnv(ref.Env)
		if !present || strings.TrimSpace(value) == "" {
			return nil, fmt.Errorf("envoy evaluation credential is unavailable")
		}
		envoyCredential = ref.Env + "=" + value
	}
	if envoyCredential != "" {
		environment = append(environment, envoyCredential)
	}
	if manifest.Target.RouterAPIKey != nil {
		return nil, fmt.Errorf("dedicated Router evaluation credentials are not supported")
	}
	return environment, nil
}

func allowlistedWorkerEnvironment() []string {
	keys := []string{
		"PATH", "PYTHONPATH", "HOME", "TMPDIR", "TEMP", "TMP",
		"LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ",
		"SSL_CERT_FILE", "SSL_CERT_DIR",
	}
	environment := make([]string, 0, len(keys))
	for _, key := range keys {
		if value, present := os.LookupEnv(key); present {
			environment = append(environment, key+"="+value)
		}
	}
	return environment
}

func decodeWorkerEvent(line []byte) (WorkerEvent, error) {
	if len(line) == 0 || len(line) > maxWorkerEventLineBytes || !utf8.Valid(line) {
		return WorkerEvent{}, fmt.Errorf("evaluation worker event exceeds the safe protocol envelope")
	}
	decoder := json.NewDecoder(bytes.NewReader(line))
	decoder.DisallowUnknownFields()
	var event WorkerEvent
	if err := decoder.Decode(&event); err != nil {
		return WorkerEvent{}, fmt.Errorf("decode evaluation worker event: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return WorkerEvent{}, err
	}
	return sanitizeWorkerEvent(event)
}

func sanitizeWorkerEvent(event WorkerEvent) (WorkerEvent, error) {
	if !allowedWorkerEventType(event.Type) {
		return WorkerEvent{}, fmt.Errorf("unknown evaluation worker event type %q", event.Type)
	}
	message := strings.TrimSpace(event.Message)
	if message == "" || len(message) > maxWorkerMessageBytes {
		return WorkerEvent{}, fmt.Errorf("evaluation worker event message is invalid")
	}
	if event.Progress != nil && len(event.Progress.Message) > maxWorkerMessageBytes {
		return WorkerEvent{}, fmt.Errorf("evaluation worker progress message is invalid")
	}
	if err := validateWorkerEventPayload(event.Type, event.Payload); err != nil {
		return WorkerEvent{}, err
	}
	event.Message = publicWorkerEventMessage(event.Type)
	if event.Progress != nil {
		progress := *event.Progress
		progress.Message = event.Message
		event.Progress = &progress
	}
	return event, nil
}

func validateWorkerEventPayload(eventType string, payload *WorkerEventPayload) error {
	if payload == nil {
		return nil
	}
	if payload.RecordCount != nil && (*payload.RecordCount < 0 || *payload.RecordCount > maxWorkerRecordCount) {
		return fmt.Errorf("evaluation worker record_count is outside the safe protocol envelope")
	}
	if payload.Verdict != "" {
		switch payload.Verdict {
		case "pass", "fail", "unavailable", "waived", "not_applicable":
		default:
			return fmt.Errorf("evaluation worker verdict is invalid")
		}
	}
	switch eventType {
	case "track":
		if payload.RecordCount == nil || payload.Verdict != "" {
			return fmt.Errorf("track worker event requires only record_count payload")
		}
	case "completed":
		if payload.RecordCount != nil || payload.Verdict == "" {
			return fmt.Errorf("completed worker event requires only verdict payload")
		}
	default:
		return fmt.Errorf("worker event type %q does not accept a payload", eventType)
	}
	return nil
}

func publicWorkerEventMessage(eventType string) string {
	switch eventType {
	case "snapshot":
		return "Evaluation snapshot validated"
	case "progress":
		return "Evaluation progress updated"
	case "track":
		return "Evaluation track evidence collected"
	case "gate":
		return "Evaluation gate updated"
	case "artifact":
		return "Evaluation artifact finalized"
	case "completed":
		return "Evaluation worker completed"
	case "failed":
		return "Evaluation worker reported failure"
	case "cancelled":
		return "Evaluation worker reported cancellation"
	default:
		return "Evaluation worker event"
	}
}

func allowedWorkerEventType(eventType string) bool {
	switch eventType {
	case "snapshot", "progress", "track", "gate", "artifact", "completed", "failed", "cancelled":
		return true
	default:
		return false
	}
}
