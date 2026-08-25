package agentruntime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

const (
	defaultLiveEventBuffer = 256
	maximumLiveEventBytes  = 128 << 10
)

type RedisLiveEventBrokerOptions struct {
	Client      redis.UniversalClient
	KeyPrefix   string
	ChannelSize int
}

// RedisLiveEventBroker provides cross-replica previews through Pub/Sub. It is
// intentionally ephemeral: session history, resume cursors, recovery, and
// correctness continue to use PostgreSQL agent_events and checkpoints.
type RedisLiveEventBroker struct {
	client      redis.UniversalClient
	keyPrefix   string
	channelSize int
	publish     chan livePublish
	cancel      context.CancelFunc
	done        chan struct{}
	closeOnce   sync.Once
}

type livePublish struct {
	channel string
	payload []byte
}

func NewRedisLiveEventBroker(options RedisLiveEventBrokerOptions) (*RedisLiveEventBroker, error) {
	prefix := strings.TrimSpace(options.KeyPrefix)
	if options.Client == nil || prefix == "" || strings.ContainsAny(prefix, "\x00\r\n *?[]") {
		return nil, errors.New("agent live event broker options are invalid")
	}
	if options.ChannelSize == 0 {
		options.ChannelSize = defaultLiveEventBuffer
	}
	if options.ChannelSize < 16 || options.ChannelSize > 4096 {
		return nil, errors.New("agent live event channel size is invalid")
	}
	brokerContext, cancel := context.WithCancel(context.Background())
	broker := &RedisLiveEventBroker{
		client: options.Client, keyPrefix: prefix, channelSize: options.ChannelSize,
		publish: make(chan livePublish, options.ChannelSize*4), cancel: cancel, done: make(chan struct{}),
	}
	go broker.runPublisher(brokerContext)
	return broker, nil
}

func (broker *RedisLiveEventBroker) PublishLiveModelStep(
	ctx context.Context, namespaceID string, value agentmanagement.LiveModelStepEvent,
) error {
	if broker == nil || broker.client == nil || uuid.Validate(namespaceID) != nil {
		return agentmanagement.ErrInvalid
	}
	normalized, err := agentmanagement.NormalizeLiveModelStepEvent(value)
	if err != nil {
		return err
	}
	payload, err := json.Marshal(normalized)
	if err != nil || len(payload) > maximumLiveEventBytes {
		return agentmanagement.ErrInvalid
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case broker.publish <- livePublish{
		channel: broker.channel(namespaceID, normalized.SessionID), payload: payload,
	}:
		return nil
	default:
		// Preview delivery may be shed under backpressure. Durable events are
		// never routed through this queue and remain available for reconcile.
		return fmt.Errorf("agent live event queue is full")
	}
}

func (broker *RedisLiveEventBroker) runPublisher(ctx context.Context) {
	defer close(broker.done)
	for {
		select {
		case <-ctx.Done():
			return
		case value := <-broker.publish:
			publishContext, cancel := context.WithTimeout(ctx, time.Second)
			_ = broker.client.Publish(publishContext, value.channel, value.payload).Err()
			cancel()
		}
	}
}

// Close stops only the broker's ephemeral publisher. The shared Redis client
// remains owned by the Management runtime composition.
func (broker *RedisLiveEventBroker) Close() error {
	if broker == nil {
		return nil
	}
	broker.closeOnce.Do(broker.cancel)
	<-broker.done
	return nil
}

func (broker *RedisLiveEventBroker) SubscribeLiveModelSteps(
	ctx context.Context, namespaceID, sessionID string,
) (agentmanagement.LiveEventSubscription, error) {
	if broker == nil || broker.client == nil || uuid.Validate(namespaceID) != nil ||
		uuid.Validate(sessionID) != nil {
		return nil, agentmanagement.ErrInvalid
	}
	subscriptionContext, cancel := context.WithCancel(ctx)
	pubsub := broker.client.Subscribe(subscriptionContext, broker.channel(namespaceID, sessionID))
	if _, err := pubsub.Receive(subscriptionContext); err != nil {
		cancel()
		_ = pubsub.Close()
		return nil, fmt.Errorf("subscribe Agent live events: %w", err)
	}
	subscription := &redisLiveEventSubscription{
		cancel: cancel, pubsub: pubsub,
		events: make(chan agentmanagement.LiveModelStepEvent, broker.channelSize),
		done:   make(chan struct{}),
	}
	go subscription.run(subscriptionContext, pubsub.Channel(
		redis.WithChannelSize(broker.channelSize),
		redis.WithChannelSendTimeout(time.Second),
	))
	return subscription, nil
}

func (broker *RedisLiveEventBroker) channel(namespaceID, sessionID string) string {
	return broker.keyPrefix + ":agent:live:" + namespaceID + ":" + sessionID
}

type redisLiveEventSubscription struct {
	cancel context.CancelFunc
	pubsub *redis.PubSub
	events chan agentmanagement.LiveModelStepEvent
	done   chan struct{}
	once   sync.Once
}

func (subscription *redisLiveEventSubscription) Events() <-chan agentmanagement.LiveModelStepEvent {
	return subscription.events
}

func (subscription *redisLiveEventSubscription) Close() error {
	if subscription == nil {
		return nil
	}
	var closeErr error
	subscription.once.Do(func() {
		subscription.cancel()
		closeErr = subscription.pubsub.Close()
	})
	return closeErr
}

func (subscription *redisLiveEventSubscription) run(
	ctx context.Context, messages <-chan *redis.Message,
) {
	defer close(subscription.done)
	defer close(subscription.events)
	defer subscription.Close()
	for {
		select {
		case <-ctx.Done():
			return
		case message, open := <-messages:
			if !open {
				return
			}
			if len(message.Payload) > maximumLiveEventBytes {
				continue
			}
			var value agentmanagement.LiveModelStepEvent
			if json.Unmarshal([]byte(message.Payload), &value) != nil {
				continue
			}
			normalized, err := agentmanagement.NormalizeLiveModelStepEvent(value)
			if err != nil {
				continue
			}
			select {
			case subscription.events <- normalized:
			case <-ctx.Done():
				return
			default:
				// Provisional output is never allowed to build an unbounded
				// queue. Ending the subscription forces the client to recover
				// from the authoritative durable stream.
				return
			}
		}
	}
}

var _ agentmanagement.LiveEventBroker = (*RedisLiveEventBroker)(nil)
