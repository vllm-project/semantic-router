package memory

// Access tracking is a documented retention contract (Memory.AccessCount and
// Memory.LastAccessed: S = S0 + AccessCount, t from LastAccessed), so every
// maintained persistent backend must reinforce the memories its Retrieve
// returns, off the request path. The assertions below fail the build when a
// backend loses its reinforcement entrypoint, and retrieveResultIDs keeps the
// result-ID collection identical across backends. The behavioral contract is
// pinned end to end by the retrieve-reinforcement contract test in the
// storage-integration lane.

// accessTrackingStore is the reinforcement entrypoint each persistent backend
// exposes for its Retrieve path.
type accessTrackingStore interface {
	recordRetrievalBatch(ids []string)
}

var (
	_ accessTrackingStore = (*MilvusStore)(nil)
	_ accessTrackingStore = (*ValkeyStore)(nil)
	_ accessTrackingStore = (*QdrantStore)(nil)
)

// retrieveResultIDs returns the IDs of the memories a Retrieve is about to
// return, in result order.
func retrieveResultIDs(results []*RetrieveResult) []string {
	ids := make([]string, len(results))
	for i, r := range results {
		ids[i] = r.Memory.ID
	}
	return ids
}
