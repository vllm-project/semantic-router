package modelservice

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func failedReplicaLease(t *testing.T, count int, failure error) (*Lease, *replicaPool) {
	t.Helper()
	manager := NewManager()
	lease := &Lease{manager: manager, members: make(map[string]member)}
	var workers []replicaWorker
	for i := range count {
		id := fmt.Sprintf("r-test-%d", i)
		plan := &processPlan{logical: "primary", key: id, name: id, replica: id, pool: true, members: map[string]string{"primary": "primary"}}
		g := newGroup(plan, nil, true)
		g.refs, g.failure, g.cancel = 1, failure, func() {}
		g.models["primary"].state = "restarting"
		close(g.done)
		manager.groups[id] = g
		lease.groups = append(lease.groups, g)
		workers = append(workers, replicaWorker{id: id, member: member{group: g, served: g.models["primary"]}})
	}
	pool := newReplicaPool(manager, "primary", config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: poolArtifact}, workers)
	lease.members["primary"] = member{group: workers[0].member.group, served: pool.served, pool: pool}
	t.Cleanup(func() { _ = lease.Close() })
	return lease, pool
}

func TestReplicaPreparationRejectsAllTerminalWorkersAndReleasesThem(t *testing.T) {
	for _, count := range []int{1, 2} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			failure := errors.New("runtime exited repeatedly before readiness")
			lease, pool := failedReplicaLease(t, count, failure)
			ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
			defer cancel()
			if err := lease.WaitManaged(ctx); !errors.Is(err, failure) || errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("terminal worker preparation waited for deadline: %v", err)
			}
			if err := lease.Close(); err != nil || len(lease.manager.groups) != 0 {
				t.Fatalf("failed generation retained its workers: %v", err)
			}
			for _, worker := range pool.workers {
				if worker.member.group.refs != 0 || worker.member.served.state != "stopped" {
					t.Fatal("discarded worker remained supervised")
				}
			}
		})
	}
}

func TestReplicaPreparationKeepsWaitingForViableWorker(t *testing.T) {
	failure := errors.New("first replica failed")
	_, pool := failedReplicaLease(t, 2, failure)
	pool.workers[1].member.group.failure = nil
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Millisecond)
	defer cancel()
	if _, err := pool.card(ctx); !errors.Is(err, context.DeadlineExceeded) || errors.Is(err, failure) {
		t.Fatalf("a pending replica must still get its readiness opportunity: %v", err)
	}
}

func TestReplicaPreparationAcceptsReadyWorkerDespiteFailedPeer(t *testing.T) {
	_, pool := failedReplicaLease(t, 2, errors.New("first replica failed"))
	worker := pool.workers[1]
	worker.member.group.failure = nil
	worker.member.group.client = &Client{}
	worker.member.group.client.bundleTasks.Store(DefaultBundleTasks)
	worker.member.served.card = &ModelCard{ID: "primary", Repo: poolArtifact, Profile: "exact", ModelSHA256: strings.Repeat("a", 64)}
	worker.member.served.state = "ready"
	worker.member.served.ready.Store(true)
	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	if card, err := pool.card(ctx); err != nil || !card.Ready {
		t.Fatalf("failed peer blocked a healthy replica: %+v %v", card, err)
	}
	worker.member.served.card.Profile = "incompatible-profile"
	if _, err := pool.card(ctx); !errors.Is(err, ErrUnavailable) || errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("failed and incompatible workers must not leave preparation pending: %v", err)
	}
}
