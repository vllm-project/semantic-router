package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
)

type blockingReplayReader struct {
	listStarted chan int
	allowList   chan struct{}
	listCalls   int
}

func (r *blockingReplayReader) Get(context.Context, string) (store.Record, bool, error) {
	return store.Record{}, false, nil
}

func (r *blockingReplayReader) List(context.Context) ([]store.Record, error) {
	r.listCalls++
	r.listStarted <- r.listCalls
	<-r.allowList
	return nil, nil
}

func TestLookupTableShutdownWaitsForInitialReplayPopulation(t *testing.T) {
	reader := &blockingReplayReader{listStarted: make(chan int), allowList: make(chan struct{})}
	cfg := &config.RouterConfig{}
	cfg.ModelSelection.LookupTables.Enabled = true
	cfg.ModelSelection.LookupTables.PopulateFromReplay = true
	_, cancel := buildLookupTable(cfg, reader)
	<-reader.listStarted

	done := make(chan struct{})
	go func() {
		cancel()
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("lookup-table shutdown returned while initial replay population was active")
	case <-time.After(50 * time.Millisecond):
	}
	reader.allowList <- struct{}{}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("lookup-table shutdown did not finish after initial replay population completed")
	}
}

func TestLookupTablePopulatorShutdownWaitsForActivePopulation(t *testing.T) {
	reader := &blockingReplayReader{listStarted: make(chan int), allowList: make(chan struct{})}
	cancel := startLookupTablePopulator(lookuptable.NewMemoryStorage(), reader, time.Millisecond)
	<-reader.listStarted
	reader.allowList <- struct{}{}
	<-reader.listStarted

	done := make(chan struct{})
	go func() {
		cancel()
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("lookup-table shutdown returned while periodic replay population was active")
	case <-time.After(50 * time.Millisecond):
	}
	reader.allowList <- struct{}{}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("lookup-table shutdown did not finish after periodic replay population completed")
	}
}
