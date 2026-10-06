package modelservice

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// processPlan is one runtime process: the models it serves and the
// deployments that call them. Its key is the process's composition, so
// router generations with the same composition share one process.
type processPlan struct {
	key      string
	name     string
	endpoint string
	models   []modelEntry
	members  map[string]string
	// threads is a CPU process's share of the router's cores.
	threads int
}

// modelEntry is one model of a --models file.
type modelEntry struct {
	Model    string `json:"model"`
	Revision string `json:"revision,omitempty"`
	Name     string `json:"name"`
	Device   string `json:"device"`
	Profile  string `json:"profile"`
}

type modelIdentity struct {
	artifact, revision, device, profile string
}

// planProcesses groups deployments into processes. Attached deployments
// share one process per endpoint and select their model by served name.
// Managed deployments share one process per process key, else per device,
// except CPU models without a key, which spread over up to
// maxCPUProcesses(cores) processes. Every CPU process runs cpuThreads threads.
// A managed deployment on auto is planned on the device auto resolves to on
// this host (empty when unknown: one process "auto"): as a cpu deployment on
// the CPU, else in that device's process, where the runtime still places it.
// Deployments of the same model, revision, device and profile in one process
// share one loaded model.
func planProcesses(deployments map[string]config.ModelDeployment, command []string, cacheDir string, cores int, auto string) []*processPlan {
	attached := make(map[string]*processPlan)
	managed := make(map[string]*processPlan)
	served := make(map[string]map[modelIdentity]string)
	names := make([]string, 0, len(deployments))
	resolved := make(map[string]config.ModelDeployment, len(deployments))
	for name, deployment := range deployments {
		names = append(names, name)
		deployment = deployment.WithDefaults()
		if deployment.Managed() && deployment.Device == autoDevice && auto == "cpu" {
			deployment.Device = "cpu"
		}
		resolved[name] = deployment
	}
	sort.Strings(names)
	add := func(process, name string, deployment config.ModelDeployment) {
		plan := managed[process]
		if plan == nil {
			plan = &processPlan{name: process, members: map[string]string{}}
			managed[process] = plan
			served[process] = make(map[modelIdentity]string)
		}
		identity := modelIdentity{deployment.Artifact, deployment.Revision, deployment.Device, deployment.Profile}
		if model, shared := served[process][identity]; shared {
			plan.members[name] = model
			return
		}
		served[process][identity] = name
		plan.members[name] = name
		plan.models = append(plan.models, modelEntry{Model: deployment.Artifact, Revision: deployment.Revision, Name: name, Device: deployment.Device, Profile: deployment.Profile})
	}
	var spread []string
	for _, name := range names {
		deployment := resolved[name]
		if endpoint := strings.TrimSpace(deployment.Endpoint); endpoint != "" {
			plan := attached[endpoint]
			if plan == nil {
				plan = &processPlan{key: "attached\x00" + endpoint, name: "attached", endpoint: endpoint, members: map[string]string{}}
				attached[endpoint] = plan
			}
			model := deployment.ServedName
			if model == "" {
				model = name
			}
			plan.members[name] = model
			continue
		}
		switch {
		case deployment.Process != "":
			add(deployment.Process, name, deployment)
		case deployment.Device == "cpu":
			spread = append(spread, name)
		case deployment.Device == autoDevice && auto != "":
			add(auto, name, deployment)
		default:
			add(deployment.Device, name, deployment)
		}
	}
	spreadCPUModels(spread, resolved, cores, add)
	shareCPUThreads(managed, cores)
	plans := make([]*processPlan, 0, len(attached)+len(managed))
	for _, plan := range managed {
		composition, _ := json.Marshal(struct {
			Models   []modelEntry
			Command  []string
			CacheDir string
			Threads  int
		}{plan.models, command, cacheDir, plan.threads})
		sum := sha256.Sum256(composition)
		plan.key = "managed\x00" + hex.EncodeToString(sum[:])
		plans = append(plans, plan)
	}
	for _, plan := range attached {
		plans = append(plans, plan)
	}
	sort.Slice(plans, func(i, j int) bool { return plans[i].key < plans[j].key })
	return plans
}

// spreadCPUModels assigns CPU models without a process key round-robin, in
// name order, to processes cpu-0 … cpu-(n-1), or to one process "cpu" when
// n is 1; deployments of one model share its process.
func spreadCPUModels(names []string, deployments map[string]config.ModelDeployment, cores int, add func(process, name string, deployment config.ModelDeployment)) {
	shard := make(map[modelIdentity]int)
	for _, name := range names {
		deployment := deployments[name].WithDefaults()
		identity := modelIdentity{deployment.Artifact, deployment.Revision, deployment.Device, deployment.Profile}
		if _, ok := shard[identity]; !ok {
			shard[identity] = len(shard)
		}
	}
	processes := min(len(shard), maxCPUProcesses(cores))
	for _, name := range names {
		deployment := deployments[name].WithDefaults()
		process := "cpu"
		if processes > 1 {
			index := shard[modelIdentity{deployment.Artifact, deployment.Revision, deployment.Device, deployment.Profile}]
			process = fmt.Sprintf("cpu-%d", index%processes)
		}
		add(process, name, deployment)
	}
}

// shareCPUThreads sizes every process whose models all run on CPU to an
// equal share of the cores.
func shareCPUThreads(managed map[string]*processPlan, cores int) {
	var onCPU []*processPlan
	for _, plan := range managed {
		cpu := len(plan.models) > 0
		for _, model := range plan.models {
			cpu = cpu && model.Device == "cpu"
		}
		if cpu {
			onCPU = append(onCPU, plan)
		}
	}
	for _, plan := range onCPU {
		plan.threads = cpuThreads(cores, len(onCPU))
	}
}

var unsafeFileName = regexp.MustCompile(`[^A-Za-z0-9_.-]+`)

// files names the socket and models file of a managed process inside dir.
func (p *processPlan) files(dir string) (socket, modelsFile string) {
	digest := strings.TrimPrefix(p.key, "managed\x00")
	if len(digest) > 12 {
		digest = digest[:12]
	}
	base := unsafeFileName.ReplaceAllString(p.name, "_") + "-" + digest
	return filepath.Join(dir, base+".sock"), filepath.Join(dir, base+".models.json")
}

// writeModelsFile writes the --models file (JSON is valid YAML) with owner-only access.
func (p *processPlan) writeModelsFile(path string) error {
	data, err := json.MarshalIndent(struct {
		Models []modelEntry `json:"models"`
	}{p.models}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o600)
}

// ManagedRequestBytes bounds a managed runtime's request bodies. The runtime
// serves the router alone over its private socket, so the bound only has to
// hold the largest request the router sends: a request stage's bundle, which
// can carry images (two at the 20 MB per-image cap of the common chat APIs,
// base64-encoded, fit).
const ManagedRequestBytes = 64 << 20

// managedBundleTasks caps the tasks of a managed runtime's bundles. A request
// stage sends each process one bundle with every call it makes, and PII alone
// makes a call per text chunk, so the cap is far above the runtime's default.
// The runtime reports it in /v1/models, and a stage with more calls is split.
const managedBundleTasks = 1024

func managedCommand(base []string, socket, modelsFile, cacheDir string, threads int) []string {
	command := append(append([]string(nil), base...), "serve", "--models", modelsFile, "--uds", socket,
		"--max-request-bytes", strconv.Itoa(ManagedRequestBytes), "--max-bundle-tasks", strconv.Itoa(managedBundleTasks))
	if cacheDir != "" {
		command = append(command, "--cache-dir", cacheDir)
	}
	if threads > 0 {
		command = append(command, "--threads", strconv.Itoa(threads))
	}
	return command
}

func (p *processPlan) deployments() []string {
	names := make([]string, 0, len(p.members))
	for name := range p.members {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// maxSocketPath keeps a Unix socket path under the kernel's sockaddr limit
// (108 bytes on Linux), which binding and dialing both enforce.
const maxSocketPath = 100

// shortSocketDir is the socket directory for a runtime directory whose
// socket paths would be too long: a private directory under /tmp named after it.
func shortSocketDir(runtimeDir string) string {
	sum := sha256.Sum256([]byte(runtimeDir))
	return filepath.Join("/tmp", "vsr-"+hex.EncodeToString(sum[:4]))
}

// privateDir creates dir with owner-only access, or tightens an existing one;
// a symbolic link is refused, so another user cannot redirect the sockets.
func privateDir(dir string) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	info, err := os.Lstat(dir)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return fmt.Errorf("%s is not a directory", dir)
	}
	return os.Chmod(dir, 0o700)
}
