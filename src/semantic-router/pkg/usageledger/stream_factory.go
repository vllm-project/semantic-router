package usageledger

import (
	"fmt"

	"github.com/redis/go-redis/v9"
)

// StreamFactory opens the partition-local stream used by one namespace worker.
// Consumer group names are shared across replicas; consumer names are unique
// to a replica and namespace.
type StreamFactory interface {
	OpenNamespaceStream(ActiveNamespace, string, string) (Stream, error)
}

// RedisStreamFactory binds namespace quota partitions to Redis streams.
type RedisStreamFactory struct {
	client    redis.UniversalClient
	keyPrefix string
}

func NewRedisStreamFactory(client redis.UniversalClient, keyPrefix string) (*RedisStreamFactory, error) {
	if client == nil {
		return nil, fmt.Errorf("usage stream Redis client is required")
	}
	if keyPrefix != "" && (len(keyPrefix) > 128 || !keyPrefixPattern.MatchString(keyPrefix)) {
		return nil, fmt.Errorf("usage stream key prefix is not canonical")
	}
	return &RedisStreamFactory{client: client, keyPrefix: keyPrefix}, nil
}

func (factory *RedisStreamFactory) OpenNamespaceStream(
	namespace ActiveNamespace,
	group, consumer string,
) (Stream, error) {
	if factory == nil || factory.client == nil {
		return nil, fmt.Errorf("usage stream factory is unavailable")
	}
	return NewRedisStream(factory.client, RedisStreamOptions{
		KeyPrefix: factory.keyPrefix,
		Partition: namespace.QuotaPartitionID,
		Group:     group,
		Consumer:  consumer,
	})
}
