# Redis Cluster Storage for Response API

This guide covers **Redis Cluster** deployment for the Response API, providing high availability, automatic failover, and data sharding across multiple nodes.

> **Note:** For simple standalone Redis setup, see [Redis Storage Guide](redis-storage.md).

## What is Redis Cluster?

**Redis Cluster** provides:

- ✅ **Data sharding**: Automatically distributes data across multiple master nodes (16384 hash slots)
- ✅ **High availability**: Automatic failover if a master fails (requires replicas)
- ✅ **Horizontal scaling**: Add more nodes to increase capacity
- ✅ **No single point of failure**: Data replicated across nodes

**vs. Standalone Redis:**

- Standalone = 1 node, simple, good for dev/small deployments
- Cluster = 6+ nodes (3 masters + 3 replicas), production-ready

## Setup and Deployment

### 1. Start Redis Cluster

#### Step 1: Create Docker Network

```bash
docker network create redis-cluster-net
```

#### Step 2: Start 6 Redis Nodes

```bash
for port in 7001 7002 7003 7004 7005 7006; do
  docker run -d \
    --name redis-node-$port \
    --network redis-cluster-net \
    -p $port:6379 \
    redis:7-alpine \
    redis-server --cluster-enabled yes \
                 --cluster-config-file nodes.conf \
                 --cluster-node-timeout 5000 \
                 --appendonly yes \
                 --port 6379
done
```

**What this does:**

- Starts 6 independent Redis servers
- Enables cluster mode on each
- Ports 7001-7006 exposed on localhost

#### Step 3: Create the Cluster

```bash
docker run --rm --network redis-cluster-net redis:7-alpine \
  redis-cli --cluster create \
  redis-node-7001:6379 \
  redis-node-7002:6379 \
  redis-node-7003:6379 \
  redis-node-7004:6379 \
  redis-node-7005:6379 \
  redis-node-7006:6379 \
  --cluster-replicas 1 --cluster-yes
```

**What this does:**

- Connects the 6 nodes into a cluster
- Creates 3 masters (7001, 7002, 7003)
- Creates 3 replicas (7004, 7005, 7006)
- Distributes hash slots: 0-5460, 5461-10922, 10923-16383

#### Step 4: Verify Cluster is Running

```bash
docker exec redis-node-7001 redis-cli cluster info
docker exec redis-node-7001 redis-cli cluster nodes
```

### 2. Configure Semantic Router

#### Option 1: Inline Configuration

Edit `config/config.yaml`:

```yaml
response_api:
  enabled: true
  store_backend: "redis"
  ttl_seconds: 86400
  redis:
    cluster_mode: true
    cluster_addresses:
      - "127.0.0.1:7001"
      - "127.0.0.1:7002"
      - "127.0.0.1:7003"
      - "127.0.0.1:7004"
      - "127.0.0.1:7005"
      - "127.0.0.1:7006"
    db: 0  # MUST be 0 for cluster
    key_prefix: "sr:"
    pool_size: 20       # Higher for cluster
    max_retries: 5      # More retries for redirects
    dial_timeout: 10    # Longer for cluster
```

#### Option 2: External Config File

Edit `config/config.yaml`:

```yaml
response_api:
  enabled: true
  store_backend: "redis"
  ttl_seconds: 86400
  redis:
    config_path: "config/response-api/redis-cluster.yaml"
```

Then edit `config/response-api/redis-cluster.yaml` with cluster addresses.

### 3. Run Semantic Router

```bash
make build-router
make run-router
```

### 4. Run EnvoyProxy

```bash
# Start Envoy proxy
make run-envoy
```

### 5. Verify Cluster Initialization

**Check logs for cluster initialization:**

```bash
tail -f /tmp/router.log | grep -i "cluster\|redis"
```

**Expected:**

```
RedisStore: creating cluster client (nodes=6, pool_size=20)
RedisStore: initialized successfully (cluster_mode=true, key_prefix=sr:, ttl=24h0m0s)
Response API enabled with redis backend
```

### 6. Test Response API

> **Note:** The examples below use `llm-katan` (Qwen3-0.6B) as the LLM backend. Adjust the `model` name to match your vLLM configuration.

#### Test 1: Create Response

```bash
curl -X POST http://localhost:8801/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "input": "What is Redis Cluster?",
    "instructions": "You are a database expert.",
    "store": true
  }'
```

**Response:**

```json
{
  "id": "resp_bb63817af32280b4a3a8fb7f",
  "object": "response",
  "status": "completed",
  "model": "qwen3",
  ...
}
```

#### Test 2: Verify Data Distribution

Check which node stores the data:

```bash
for port in 7001 7002 7003 7004 7005 7006; do
  echo "=== Node $port ==="
  docker exec redis-node-$port redis-cli KEYS "sr:*"
done
```

**Example output:**

```
=== Node 7001 ===
sr:response:resp_bb63817af32280b4a3a8fb7f
=== Node 7002 ===

=== Node 7003 ===

=== Node 7004 ===

=== Node 7005 ===
sr:response:resp_bb63817af32280b4a3a8fb7f  # Replica of 7001
=== Node 7006 ===
```

**This shows:**

- Master 7001 has the data (hash slot matched)
- Replica 7005 has a copy (backup)
- Other nodes are empty (different hash slots)

#### Test 3: Retrieve Response

```bash
curl -X GET http://localhost:8801/v1/responses/resp_bb63817af32280b4a3a8fb7f
```

**The client automatically:**

- Calculates hash slot for the key
- Routes request to correct node (7001)
- Handles MOVED redirects if needed

#### Test 4: Conversation Chaining

```bash
curl -X POST http://localhost:8801/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "input": "Tell me more about sharding",
    "previous_response_id": "resp_bb63817af32280b4a3a8fb7f",
    "store": true
  }'
```

**Response:**

```json
{
  "id": "resp_a4ae205a80ae7bf10edecaa3",
  "previous_response_id": "resp_bb63817af32280b4a3a8fb7f",
  "status": "completed",
  ...
}
```

#### Test 5: Delete Response

```bash
curl -X DELETE http://localhost:8801/v1/responses/resp_bb63817af32280b4a3a8fb7f
```

**Deletion works across cluster:**

- Client finds correct node
- Deletes from master
- Replica syncs automatically

## Cluster Monitoring

### Check Cluster Health

```bash
docker exec redis-node-7001 redis-cli cluster info
```

**Key metrics:**

- `cluster_state:ok` - Cluster is healthy
- `cluster_slots_assigned:16384` - All slots assigned
- `cluster_known_nodes:6` - All nodes discovered

### View Node Roles

```bash
docker exec redis-node-7001 redis-cli cluster nodes
```

**Output shows:**

- Master nodes with hash slot ranges
- Replica nodes and which master they backup

### Monitor Keys per Node

```bash
for port in 7001 7002 7003; do
  count=$(docker exec redis-node-$port redis-cli DBSIZE)
  echo "Master $port: $count keys"
done
```

## Cleanup

### Stop and Remove All Nodes

```bash
for port in 7001 7002 7003 7004 7005 7006; do
  docker stop redis-node-$port
  docker rm redis-node-$port
done

docker network rm redis-cluster-net
```

## Reference

- [Redis Storage (Standalone)](redis-storage.md) - Simple standalone setup
- Configuration: `config/response-api/redis-cluster.yaml`
- Integration tests: `pkg/responsestore/redis_store_integration_test.go`
