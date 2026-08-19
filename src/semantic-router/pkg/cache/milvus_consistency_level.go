package cache

import (
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	milvuslifecycle "github.com/vllm-project/semantic-router/src/semantic-router/pkg/milvus"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// resolveMilvusConsistencyLevel maps a configured consistency level name to
// the SDK constant via the shared parser. ok is false when the name is empty
// or unrecognized; the caller then leaves the option unset so the Milvus SDK
// default (Bounded) stays in effect, preserving today's behavior for
// deployments that do not pin a level.
func resolveMilvusConsistencyLevel(name string) (entity.ConsistencyLevel, bool) {
	return milvuslifecycle.ParseConsistencyLevel(name)
}

// searchQueryOptions returns the read options that pin every Milvus search
// and query the cache issues against its own collection to the configured
// consistency level. Reads against foreign collections (e.g. RAG knowledge
// bases via SearchDocuments) must not use these options: without an explicit
// option the SDK adopts the target collection's server-side level, which is
// the intended behavior there. An empty or unrecognized level yields no
// options, so the SDK default applies and the behavior of existing
// deployments is unchanged.
func (c *MilvusCache) searchQueryOptions() []client.SearchQueryOptionFunc {
	level, ok := resolveMilvusConsistencyLevel(c.config.Search.ConsistencyLevel)
	if !ok {
		return nil
	}
	return []client.SearchQueryOptionFunc{client.WithSearchQueryConsistencyLevel(level)}
}

// warnUnrecognizedMilvusConsistencyLevel logs a warning when a configured
// consistency level name is not one of the supported values, so a typo
// surfaces instead of silently keeping the SDK default. An empty value is
// the documented "use the SDK default" case and is not warned about.
func warnUnrecognizedMilvusConsistencyLevel(name string) {
	if name == "" {
		return
	}
	if _, ok := resolveMilvusConsistencyLevel(name); ok {
		return
	}
	logging.Warnf("MilvusCache: unrecognized consistency_level %q (valid: %s); leaving the Milvus SDK default in effect", name, milvuslifecycle.ConsistencyLevelNames)
}

// createCollectionOptions returns the collection-creation options that pin a
// newly created collection to the configured consistency level, mirroring
// searchQueryOptions on the schema-creation path.
func (c *MilvusCache) createCollectionOptions() []client.CreateCollectionOption {
	level, ok := resolveMilvusConsistencyLevel(c.config.Search.ConsistencyLevel)
	if !ok {
		return nil
	}
	return []client.CreateCollectionOption{client.WithConsistencyLevel(level)}
}
