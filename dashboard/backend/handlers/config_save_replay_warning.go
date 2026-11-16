package handlers

import (
	"fmt"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// replayStoreResetCode marks a save after which the Router starts the memory
// store that held Router Replay's records empty.
const replayStoreResetCode = "replay_store_reset"

// replayStoreResetWarnings warns about a save that loses records Router
// Replay keeps in memory. A reload keeps a memory store only while the records
// stay in memory with the same ttl_seconds and a recorder of the same name,
// as RouterReplayMemoryStores names them, still exists.
func replayStoreResetWarnings(previous []byte, next *routerconfig.RouterConfig) []routerconfig.ConfigWarning {
	if next == nil || len(previous) == 0 {
		return nil
	}
	before, err := routerconfig.ParseYAMLBytesDeferringEnv(previous)
	if err != nil || before.RouterReplayStoreBackend() != "memory" {
		return nil
	}
	warning := routerconfig.ConfigWarning{Code: replayStoreResetCode}
	switch backend := next.RouterReplayStoreBackend(); {
	case backend != "memory":
		warning.Field = "global.services.router_replay.store_backend"
		warning.Message = fmt.Sprintf("Router Replay kept its records in memory and keeps them in the %s store after this save: the records kept so far are lost.", backend)
	case next.RouterReplay.TTLSeconds != before.RouterReplay.TTLSeconds:
		warning.Field = "global.services.router_replay.ttl_seconds"
		warning.Message = "Router Replay keeps its records in memory, and a changed ttl_seconds starts that store empty: the records kept so far are lost."
	default:
		lost := droppedReplayStores(before, next)
		if len(lost) == 0 {
			return nil
		}
		warning.Field = "global.services.router_replay"
		warning.Message = fmt.Sprintf("Router Replay keeps the records of %s in a memory store of its own, which this save no longer has, such as after a rename: those records are lost.", strings.Join(lost, ", "))
	}
	return []routerconfig.ConfigWarning{warning}
}

// droppedReplayStores names the memory stores before has and next does not.
func droppedReplayStores(before, next *routerconfig.RouterConfig) []string {
	kept := map[string]bool{}
	for _, store := range next.RouterReplayMemoryStores() {
		kept[store.Name] = true
	}
	var lost []string
	for _, store := range before.RouterReplayMemoryStores() {
		if !kept[store.Name] {
			lost = append(lost, store.Name)
		}
	}
	return lost
}
