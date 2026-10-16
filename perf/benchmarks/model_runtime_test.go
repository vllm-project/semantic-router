//go:build !windows

package benchmarks

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/perf/pkg/benchmark"
	"github.com/vllm-project/semantic-router/perf/pkg/modelassets"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// benchmarkInput is the shared deployment budget: 512 tokens, truncating.
var benchmarkInput = config.ModelInputBudget{MaxTokens: 512, Overflow: "truncate"}

// Benchmarks reach every model the way the router does: through one runtime
// manager and the serving facade, each deployment in a runtime process of its
// own (VLLM_SRUN_COMMAND, else vllm-srun on PATH).
var (
	benchmarkManager    = modelservice.NewManager()
	benchmarkLease      = acquireBenchmarkLease()
	benchmarkRuntime    = serving.New(benchmarkLease, nil)
	benchmarkArtifacts  = map[string]benchmark.ModelArtifact{}
	benchmarkCards      = map[string]modelservice.ModelCard{}
	benchmarkSpecs      = map[string]config.ResolvedModelBinding{}
	benchmarkIdentities = map[string]benchmark.ModelIdentity{}
	benchmarkModelMu    sync.Mutex
)

func acquireBenchmarkLease() *modelservice.Lease {
	lease, err := benchmarkManager.AcquireDeployments(nil)
	if err != nil {
		panic(fmt.Sprintf("model runtime manager: %v", err))
	}
	return lease
}

func benchmarkModel(b *testing.B, name, contract string) config.ResolvedModelBinding {
	return benchmarkDeployment(b, name, contract, "perf-"+name, benchmarkInput)
}

// benchmarkDeployment starts a deployment of a catalog model once and records
// the identity of the package the runtime actually loaded.
func benchmarkDeployment(b *testing.B, name, contract, deployment string, input config.ModelInputBudget) config.ResolvedModelBinding {
	b.Helper()
	benchmarkModelMu.Lock()
	defer benchmarkModelMu.Unlock()
	if spec, ok := benchmarkSpecs[deployment]; ok {
		return spec
	}
	root, err := filepath.Abs("../..")
	if err != nil {
		b.Fatal(err)
	}
	artifact, err := modelassets.Resolve(name, root)
	if err != nil {
		b.Fatal(err)
	}
	definition := artifact.Deployment(cacheEmbeddingDevice(), input)
	if err = benchmarkLease.Ensure(deployment, definition); err != nil {
		b.Fatalf("start the %s model runtime: %v", name, err)
	}
	card, err := benchmarkLease.Card(context.Background(), deployment)
	if err != nil {
		b.Fatalf("the %s model runtime is not ready: %v", name, err)
	}
	repo, revision := card.Repo, card.Revision
	if repo == "" {
		repo, revision = artifact.RepoID, artifact.Revision
	}
	benchmarkArtifacts[name] = benchmark.ModelArtifact{RepoID: repo, Revision: revision, ContentsSHA256: card.ModelSHA256}
	benchmarkCards[name] = card
	spec := config.ResolvedModelBinding{
		Recipe: "perf", Name: name,
		Binding:    config.ModelBinding{Deployment: deployment, Contract: contract},
		Deployment: definition,
	}
	benchmarkSpecs[deployment] = spec
	return spec
}

func recordModelIdentity(b *testing.B, names ...string) {
	benchmarkModelMu.Lock()
	defer benchmarkModelMu.Unlock()
	artifacts := map[string]benchmark.ModelArtifact{}
	for _, name := range names {
		artifacts[name] = benchmarkArtifacts[name]
	}
	card := benchmarkCards[names[0]]
	protocol := fmt.Sprintf("model-runtime-v1;engine=%s;profile=%s;max_tokens=512;overflow=truncate;embedding=full-layer/full-dimension", card.Engine, card.Profile)
	if _, ok := artifacts["embedding"]; ok {
		// InMemoryCache applies these options to every mmBERT provider; this
		// describes the measured workload, independent of owner setup.
		protocol = fmt.Sprintf("model-runtime-v1;engine=%s;profile=%s;max_tokens=512;overflow=truncate;embedding=layer-6/dimension-256", card.Engine, card.Profile)
	}
	benchmarkIdentities[b.Name()] = benchmark.ModelIdentity{Artifacts: artifacts, Provider: config.ModelRuntimeProvider, Device: card.Device, Precision: card.Dtype, Protocol: protocol}
}

func recordCacheProtocol(b *testing.B, scenario cacheScenario) {
	benchmarkModelMu.Lock()
	defer benchmarkModelMu.Unlock()
	identity := benchmarkIdentities[b.Name()]
	identity.Protocol += fmt.Sprintf(";cache=public-lookup-v3;graphs=%d;op=%d-requests;corpus=20260919;memo=warm",
		scenario.samples(), scenario.requestsPerOp())
	benchmarkIdentities[b.Name()] = identity
}

func closeBenchmarkOwners(code int, owners ...io.Closer) (int, error) {
	var errs []error
	for _, owner := range owners {
		if owner != nil {
			errs = append(errs, owner.Close())
		}
	}
	err := errors.Join(errs...)
	if err != nil && code == 0 {
		code = 1
	}
	return code, err
}

func TestMain(m *testing.M) {
	code := m.Run()
	var owners []io.Closer
	if benchClassifier != nil {
		owners = append(owners, benchClassifier)
	}
	if domainTask != nil {
		owners = append(owners, domainTask)
	}
	if inputLengthDomain != nil {
		owners = append(owners, inputLengthDomain)
	}
	if cacheEmbeddingOwner != nil {
		owners = append(owners, cacheEmbeddingOwner)
	}
	owners = append(owners, benchmarkLease, runtimeShutdown{benchmarkManager})
	code, err := closeBenchmarkOwners(code, owners...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "close benchmark model owners: %v\n", err)
	}
	names := make([]string, 0, len(benchmarkIdentities))
	for name := range benchmarkIdentities {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		data, err := json.Marshal(benchmarkIdentities[name])
		if err != nil {
			panic(err)
		}
		fmt.Printf("%s%s %s\n", benchmark.ModelIdentityPrefix, name, data)
	}
	os.Exit(code)
}

// runtimeShutdown stops the runtime processes after their owners close.
type runtimeShutdown struct{ manager *modelservice.Manager }

func (r runtimeShutdown) Close() error { return r.manager.Shutdown(context.Background()) }
