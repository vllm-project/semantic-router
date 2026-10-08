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

// processPlan is one logical worker. Its identity depends on the deployment
// and placement, not the set of other consumers in a Router generation.
type processPlan struct {
	logical  string
	pool     bool
	replica  string
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

// planProcesses assigns every logical deployment worker its own process. The
// key depends only on that worker, never on other consumers or Router mode.
// Attached owners may serve multiple models; the Router does not regroup them.
func planProcesses(deployments map[string]config.ModelDeployment, command []string, cacheDir string, cores int, auto string) []*processPlan {
	var plans []*processPlan
	for name, declaration := range deployments {
		deployment := declaration.WithDefaults()
		occurrences := make(map[string]int)
		for _, placement := range deployment.Placements() {
			encoded, _ := json.Marshal(placement)
			placementKey := string(encoded)
			ordinal := occurrences[placementKey]
			occurrences[placementKey]++
			sum := sha256.Sum256([]byte(fmt.Sprintf("%s\x00%s\x00%d", name, placementKey, ordinal)))
			id := "r-" + hex.EncodeToString(sum[:6])
			memberName := name
			if len(deployment.Replicas) > 0 {
				memberName += "\x00" + id
			}
			servedName := placement.ServedName
			if servedName == "" {
				servedName = name
			}
			plan := &processPlan{logical: name, pool: len(deployment.Replicas) > 0, replica: id, name: name + "-" + id, endpoint: placement.Endpoint, members: map[string]string{memberName: servedName}}
			if placement.Endpoint == "" {
				device := placement.Device
				if device == autoDevice && auto == "cpu" {
					device = "cpu"
				}
				plan.models = []modelEntry{{Model: deployment.Artifact, Revision: deployment.Revision, Name: name, Device: device, Profile: deployment.Profile}}
				if device == "cpu" {
					plan.threads = cpuThreads(cores, maxCPUProcesses(cores))
				}
			}
			identity, _ := json.Marshal(struct {
				Logical, Replica, Endpoint, Served string
				Models                             []modelEntry
				Command                            []string
				CacheDir                           string
				Threads                            int
			}{name, id, placement.Endpoint, servedName, plan.models, command, cacheDir, plan.threads})
			digest := sha256.Sum256(identity)
			prefix := "managed"
			if placement.Endpoint != "" {
				prefix = "attached"
			}
			plan.key = prefix + "\x00" + hex.EncodeToString(digest[:])
			plans = append(plans, plan)
		}
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
