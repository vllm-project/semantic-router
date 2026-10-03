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
// Managed deployments share one process per process key, else per device;
// deployments of the same model, revision, device and profile in one
// process share one loaded model.
func planProcesses(deployments map[string]config.ModelDeployment, command []string, cacheDir string) []*processPlan {
	attached := make(map[string]*processPlan)
	managed := make(map[string]*processPlan)
	served := make(map[string]map[modelIdentity]string)
	names := make([]string, 0, len(deployments))
	for name := range deployments {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		deployment := deployments[name].WithDefaults()
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
		process := deployment.Process
		if process == "" {
			process = deployment.Device
		}
		plan := managed[process]
		if plan == nil {
			plan = &processPlan{name: process, members: map[string]string{}}
			managed[process] = plan
			served[process] = make(map[modelIdentity]string)
		}
		identity := modelIdentity{deployment.Artifact, deployment.Revision, deployment.Device, deployment.Profile}
		if model, shared := served[process][identity]; shared {
			plan.members[name] = model
			continue
		}
		served[process][identity] = name
		plan.members[name] = name
		plan.models = append(plan.models, modelEntry{Model: deployment.Artifact, Revision: deployment.Revision, Name: name, Device: deployment.Device, Profile: deployment.Profile})
	}
	plans := make([]*processPlan, 0, len(attached)+len(managed))
	for _, plan := range managed {
		composition, _ := json.Marshal(struct {
			Models   []modelEntry
			Command  []string
			CacheDir string
		}{plan.models, command, cacheDir})
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

func managedCommand(base []string, socket, modelsFile, cacheDir string) []string {
	command := append(append([]string(nil), base...), "serve", "--models", modelsFile, "--uds", socket)
	if cacheDir != "" {
		command = append(command, "--cache-dir", cacheDir)
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
