package modelservice

import (
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

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
	worker.served.card = &ModelCard{ID: "primary", Repo: poolArtifact, Profile: "exact", ModelSHA256: strings.Repeat("a", 64)}
	worker.served.ready.Store(true)
	if _, ok := lease.CurrentCard("primary"); !ok {
		t.Fatal("offline peer masked a compatible worker")
	}
	worker.served.card = &ModelCard{ID: "primary", Repo: "different/model", Profile: "exact", ModelSHA256: strings.Repeat("a", 64)}
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
		worker.served.card = &ModelCard{ID: "primary", Repo: poolArtifact, Profile: "exact", ModelSHA256: strings.Repeat("a", 64)}
		worker.served.ready.Store(true)
		worker.group.mu.Unlock()
	}
	_ = lease.Close()
	readers.Wait()
	if _, ok := lease.CurrentCard("primary"); ok {
		t.Fatal("closed generation leaked a concurrently observed card")
	}
}
