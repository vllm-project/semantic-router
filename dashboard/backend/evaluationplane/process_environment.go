package evaluationplane

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

type workerBrokerCredentials struct {
	router                     string
	envoy                      string
	agentTaskLedger            string
	faultRecoveryLedger        string
	hardPolicyLedger           string
	productionExperimentLedger string
}

type controlledPairCredentialFreezer interface {
	freezeControlledPairCredentials(context.Context, RunManifest) (workerBrokerCredentials, error)
}

func (p *CommandProcess) freezeControlledPairCredentials(
	ctx context.Context,
	manifest RunManifest,
) (workerBrokerCredentials, error) {
	if err := ctx.Err(); err != nil {
		return workerBrokerCredentials{}, err
	}
	return p.brokerCredentials(manifest)
}

func (p *CommandProcess) brokerCredentials(manifest RunManifest) (workerBrokerCredentials, error) {
	router, err := p.routerCredential(manifest)
	if err != nil {
		return workerBrokerCredentials{}, err
	}
	envoy, err := p.envoyCredential(manifest)
	if err != nil {
		return workerBrokerCredentials{}, err
	}
	agentTask, err := endpointCredential(manifest.Target.AgentTaskLedger, "agent-task ledger")
	if err != nil {
		return workerBrokerCredentials{}, err
	}
	faultRecovery, err := endpointCredential(manifest.Target.FaultRecoveryLedger, "fault-recovery ledger")
	if err != nil {
		return workerBrokerCredentials{}, err
	}
	hardPolicy, err := endpointCredential(manifest.Target.HardPolicyLedger, "hard-policy ledger")
	if err != nil {
		return workerBrokerCredentials{}, err
	}
	productionExperiment, err := endpointCredential(manifest.Target.ProductionExperimentLedger, "production experiment ledger")
	if err != nil {
		return workerBrokerCredentials{}, err
	}
	return workerBrokerCredentials{
		router: router, envoy: envoy, agentTaskLedger: agentTask,
		faultRecoveryLedger: faultRecovery, hardPolicyLedger: hardPolicy,
		productionExperimentLedger: productionExperiment,
	}, nil
}

func (p *CommandProcess) routerCredential(manifest RunManifest) (string, error) {
	ref := manifest.Target.RouterAPIKey
	if ref == nil {
		return "", nil
	}
	if p.routerAPIKeyEnv == "" || ref.Env != p.routerAPIKeyEnv {
		return "", fmt.Errorf("staged evaluation manifest has an unsupported Router credential reference")
	}
	value, present := os.LookupEnv(ref.Env)
	if !present || strings.TrimSpace(value) == "" {
		return "", fmt.Errorf("router evaluation credential is unavailable")
	}
	return value, nil
}

func endpointCredential(endpoint *ServiceEndpoint, label string) (string, error) {
	if endpoint == nil || endpoint.APIKey == nil {
		return "", nil
	}
	value, present := os.LookupEnv(endpoint.APIKey.Env)
	if !present || strings.TrimSpace(value) == "" {
		return "", fmt.Errorf("%s evaluation credential is unavailable", label)
	}
	return value, nil
}

func (p *CommandProcess) envoyCredential(manifest RunManifest) (string, error) {
	if ref := manifest.Target.EnvoyAPIKey; ref != nil {
		if p.envoyAPIKeyEnv != "" && ref.Env != p.envoyAPIKeyEnv {
			return "", fmt.Errorf("staged evaluation manifest has an unsupported Envoy credential reference")
		}
		value, present := os.LookupEnv(ref.Env)
		if !present || strings.TrimSpace(value) == "" {
			return "", fmt.Errorf("envoy evaluation credential is unavailable")
		}
		return value, nil
	}
	return "", nil
}

func isolatedWorkerEnvironment(sandboxRoot string) []string {
	keys := []string{"LANG", "LC_ALL", "LC_CTYPE", "TZ"}
	environment := make([]string, 0, len(keys)+7)
	for _, key := range keys {
		if value, present := os.LookupEnv(key); present {
			environment = append(environment, key+"="+value)
		}
	}
	environment = append(environment,
		"HOME="+filepath.Join(sandboxRoot, "home"),
		"TMPDIR="+filepath.Join(sandboxRoot, "tmp"),
		"TEMP="+filepath.Join(sandboxRoot, "tmp"),
		"TMP="+filepath.Join(sandboxRoot, "tmp"),
		"PYTHONUNBUFFERED=1",
		"PYTHONDONTWRITEBYTECODE=1",
		"PYTHONHASHSEED=0",
	)
	return environment
}

func resolveWorkerPython(configured string) (string, error) {
	resolved, err := exec.LookPath(configured)
	if err != nil {
		return "", fmt.Errorf("resolve evaluation worker interpreter: %w", err)
	}
	absolute, err := filepath.Abs(resolved)
	if err != nil {
		return "", fmt.Errorf("resolve evaluation worker interpreter path: %w", err)
	}
	resolvedTarget, err := filepath.EvalSymlinks(absolute)
	if err != nil {
		return "", fmt.Errorf("resolve evaluation worker interpreter target: %w", err)
	}
	info, err := os.Lstat(resolvedTarget)
	if err != nil || !info.Mode().IsRegular() {
		return "", fmt.Errorf("evaluation worker interpreter is not a regular file")
	}
	// Python discovers a virtual environment from the configured path, so keep
	// that path after validating its final regular-file target.
	return absolute, nil
}

func (p *CommandProcess) resolveWorkerScript() (string, error) {
	if p.workerScriptPath != "" {
		return validateWorkerScript(p.workerScriptPath)
	}
	seen := make(map[string]bool)
	for _, root := range filepath.SplitList(os.Getenv("PYTHONPATH")) {
		if strings.TrimSpace(root) == "" {
			continue
		}
		candidate := filepath.Join(root, filepath.FromSlash(workerSandboxScript))
		if path, err := validateWorkerScript(candidate); err == nil {
			seen[path] = true
		}
	}
	if len(seen) != 1 {
		return "", fmt.Errorf("evaluation worker sandbox entrypoint is not uniquely installed")
	}
	for path := range seen {
		return path, nil
	}
	return "", fmt.Errorf("evaluation worker sandbox entrypoint is unavailable")
}

func validateWorkerScript(raw string) (string, error) {
	absolute, err := filepath.Abs(strings.TrimSpace(raw))
	if err != nil {
		return "", err
	}
	info, err := os.Lstat(absolute)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return "", fmt.Errorf("evaluation worker sandbox entrypoint is not a regular file")
	}
	if runtime.GOOS == "linux" && info.Mode().Perm()&0o022 != 0 {
		return "", fmt.Errorf("evaluation worker sandbox entrypoint is writable by an untrusted principal")
	}
	return absolute, nil
}

func workerCPULimit(timeout time.Duration) int {
	seconds := math.Ceil(timeout.Seconds())
	if seconds < 1 {
		return 1
	}
	if seconds > float64(math.MaxInt32) {
		return math.MaxInt32
	}
	return int(seconds)
}
