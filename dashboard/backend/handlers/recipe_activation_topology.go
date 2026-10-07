package handlers

import "context"

// environmentRuntimeTopology reads the managed stack's topology from what
// `vllm-sr serve` passed the Dashboard when it created it. The Dashboard holds
// no container runtime: an activation that needs containers created anew
// waits for the next `vllm-sr serve` as a pending activation.
type environmentRuntimeTopology struct{}

func (environmentRuntimeTopology) Inventory(context.Context) (runtimeTopologyInventory, error) {
	storage, err := environmentStorageInventory()
	return runtimeTopologyInventory{Storage: storage}, err
}
