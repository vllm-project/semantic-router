package responseapirediscluster

import "github.com/vllm-project/semantic-router/e2e/pkg/profiles/responseapi"

const (
	valuesFile           = "e2e/profiles/response-api-redis-cluster/values.yaml"
	redisClusterManifest = "deploy/kubernetes/response-api/redis-cluster.yaml"
)

// Profile implements the Response API Redis Cluster test profile.
type Profile struct {
	*responseapi.RedisProfile
}

// NewProfile creates a new Response API Redis Cluster profile.
func NewProfile() *Profile {
	return &Profile{
		RedisProfile: responseapi.NewRedisProfile(
			"response-api-redis-cluster",
			"Tests Response API endpoints using Redis Cluster storage backend",
			valuesFile,
			redisClusterManifest,
		),
	}
}
