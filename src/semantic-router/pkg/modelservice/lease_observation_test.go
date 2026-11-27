package modelservice

import (
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

func TestCurrentCardReportsEligibleWorkerPrecision(t *testing.T) {
	lease, _ := testReplicaPool(t, runtimetest.Model{ID: "a", Dtype: "fp32"}, runtimetest.Model{ID: "b", Dtype: "fp32"})
	setWorker := func(i int, precision string, ready bool) {
		w := lease.members["primary"].pool.workers[i].member
		w.group.mu.Lock()
		defer w.group.mu.Unlock()
		card := *w.served.card
		card.Dtype = precision
		w.served.card = &card
		w.served.ready.Store(ready)
	}
	check := func(want string) {
		t.Helper()
		card, ok := lease.CurrentCard("primary")
		if !ok || card.Dtype != want {
			t.Fatalf("observed precision = %q (ready=%t), want %q", card.Dtype, ok, want)
		}
	}
	check("fp32")
	initial, _ := lease.CurrentCard("primary")
	setWorker(1, "", true)
	check("") // An eligible worker did not report a precision.
	setWorker(1, "bf16/fp32-readout", true)
	check("")
	setWorker(0, "fp32", false)
	check("bf16/fp32-readout") // Only the non-baseline worker can now serve.
	setWorker(0, "bf16/fp32-readout", true)
	check("bf16/fp32-readout")
	setWorker(0, "fp32", true)
	setWorker(1, "fp32", true)
	check("fp32")
	restored, _ := lease.CurrentCard("primary")
	if initial.PrecisionEpoch == 0 || restored.PrecisionEpoch == initial.PrecisionEpoch {
		t.Fatal("returning to the initial precision must not validate windows planned before the change")
	}
}

func TestCurrentCardTracksReadinessAndGenerationRetirement(t *testing.T) {
	lease, _ := testReplicaPool(t, runtimetest.Model{ID: "a"})
	observed := lease.members["primary"].pool.workers[0].member
	if _, ok := lease.CurrentCard("primary"); !ok {
		t.Fatal("ready worker has no observed card")
	}
	observed.group.mu.Lock()
	observed.served.ready.Store(false)
	observed.group.mu.Unlock()
	if _, ok := lease.CurrentCard("primary"); ok {
		t.Fatal("unready worker retained a usable card")
	}
	observed.group.mu.Lock()
	replacement := *observed.served.card
	replacement.QuestionTypes = []string{"span"}
	observed.served.card = &replacement
	observed.served.ready.Store(true)
	observed.group.mu.Unlock()
	if card, ok := lease.CurrentCard("primary"); !ok || !card.Answers("span") || card.Answers("noul") {
		t.Fatalf("recovered worker retained old capabilities: %+v %v", card, ok)
	}
	if _, ok := lease.CurrentCard("unknown"); ok {
		t.Fatal("unknown deployment inherited another worker's metadata")
	}
	_ = lease.Close()
	if _, ok := lease.CurrentCard("primary"); ok {
		t.Fatal("retired generation retained its observation")
	}
}

func TestCurrentCardRejectsIncompatibleReplicaAndConcurrentRetirement(t *testing.T) {
	lease, pool := failedReplicaLease(t, 2, errors.New("offline"))
	worker := pool.workers[1].member
	worker.group.failure = nil
	worker.group.client = &Client{}
	worker.group.client.bundleTasks.Store(DefaultBundleTasks)
	worker.served.card = &ModelCard{ID: "primary", Repo: poolArtifact, ModelSHA256: strings.Repeat("a", 64), EngineArgs: EngineArgs{Dtype: "auto"}}
	worker.served.ready.Store(true)
	if _, ok := lease.CurrentCard("primary"); !ok {
		t.Fatal("offline peer masked a compatible worker")
	}
	worker.served.card = &ModelCard{ID: "primary", Repo: "different/model", ModelSHA256: strings.Repeat("a", 64)}
	if _, ok := lease.CurrentCard("primary"); ok {
		t.Fatal("incompatible replica exposed old pool metadata")
	}
	var readers sync.WaitGroup
	for range 8 {
		readers.Go(func() {
			for range 100 {
				lease.CurrentCard("primary")
			}
		})
	}
	for range 100 {
		worker.group.mu.Lock()
		worker.served.ready.Store(false)
		worker.served.card = &ModelCard{ID: "primary", Repo: poolArtifact, ModelSHA256: strings.Repeat("a", 64), EngineArgs: EngineArgs{Dtype: "auto"}}
		worker.served.ready.Store(true)
		worker.group.mu.Unlock()
	}
	_ = lease.Close()
	readers.Wait()
	if _, ok := lease.CurrentCard("primary"); ok {
		t.Fatal("closed generation leaked a concurrently observed card")
	}
}
