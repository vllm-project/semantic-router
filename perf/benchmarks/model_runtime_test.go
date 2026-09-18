//go:build !windows && cgo

package benchmarks

import (
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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

var (
	benchmarkRuntime    = native.New(nil)
	benchmarkArtifacts  = map[string]benchmark.ModelArtifact{}
	benchmarkSpecs      = map[string]config.ResolvedModelBinding{}
	benchmarkIdentities = map[string]benchmark.ModelIdentity{}
	benchmarkModelMu    sync.Mutex
)

func benchmarkModel(b *testing.B, name, contract string) config.ResolvedModelBinding {
	b.Helper()
	benchmarkModelMu.Lock()
	defer benchmarkModelMu.Unlock()
	if spec, ok := benchmarkSpecs[name]; ok {
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
	digest, err := modelassets.ContentsSHA256(artifact.Path)
	if err != nil {
		b.Fatalf("required %s model at %s: %v; run make download-models-perf", name, artifact.Path, err)
	}
	benchmarkArtifacts[name] = benchmark.ModelArtifact{RepoID: artifact.RepoID, Revision: artifact.Revision, ContentsSHA256: digest}
	spec := config.ResolvedModelBinding{
		Recipe: "perf", Name: name,
		Binding:    config.ModelBinding{Deployment: "perf-" + name, Contract: contract, Adapter: "mmbert"},
		Deployment: config.ModelDeployment{Artifact: artifact.Path, Revision: artifact.Revision, Provider: "candle", Device: cacheEmbeddingDevice(), Precision: "fp32", Input: config.ModelInputBudget{MaxTokens: 512, Overflow: "truncate"}},
	}
	benchmarkSpecs[name] = spec
	return spec
}

func recordModelIdentity(b *testing.B, names ...string) {
	benchmarkModelMu.Lock()
	defer benchmarkModelMu.Unlock()
	artifacts := map[string]benchmark.ModelArtifact{}
	for _, name := range names {
		artifacts[name] = benchmarkArtifacts[name]
	}
	protocol := "owned-native-v1;max_tokens=512;overflow=truncate;embedding=full-layer/full-dimension"
	if _, ok := artifacts["embedding"]; ok {
		// InMemoryCache applies these options to every mmBERT provider; this
		// describes the existing measured workload, independent of owner setup.
		protocol = "owned-native-v1;max_tokens=512;overflow=truncate;embedding=layer-6/dimension-256"
	}
	benchmarkIdentities[b.Name()] = benchmark.ModelIdentity{Artifacts: artifacts, Provider: "candle", Device: cacheEmbeddingDevice(), Precision: "float32", Protocol: protocol}
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
	if cacheEmbeddingOwner != nil {
		owners = append(owners, cacheEmbeddingOwner)
	}
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
