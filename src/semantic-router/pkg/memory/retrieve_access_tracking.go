package memory

// Access tracking feeds retention scoring (Memory.AccessCount and
// Memory.LastAccessed: S = S0 + AccessCount, t from LastAccessed) for the
// persistent backends that support retrieve-time reinforcement. The assertions
// below keep the Qdrant and Valkey entrypoints present, and retrieveResultIDs
// keeps their result-ID collection identical. The behavioral contract is pinned
// end to end by the storage-integration tests.

// accessTrackingStore is the reinforcement entrypoint each reinforcing backend
// exposes for its Retrieve path.
type accessTrackingStore interface {
	recordRetrievalBatch(ids []string)
}

var (
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
