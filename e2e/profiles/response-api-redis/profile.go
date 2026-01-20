package responseapiredis

import "github.com/vllm-project/semantic-router/e2e/pkg/profiles/responseapi"

const (
	valuesFile    = "e2e/profiles/response-api-redis/values.yaml"
	redisManifest = "deploy/kubernetes/response-api/redis.yaml"
)

// Profile implements the Response API Redis test profile.
type Profile struct {
	*responseapi.RedisProfile
}

// NewProfile creates a new Response API Redis profile.
func NewProfile() *Profile {
	return &Profile{
		RedisProfile: responseapi.NewRedisProfile(
			"response-api-redis",
			"Tests Response API endpoints using Redis storage backend",
			valuesFile,
			redisManifest,
		),
	}
}
