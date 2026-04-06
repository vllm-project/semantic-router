package cache

import (
	"context"
	"crypto/md5"
	"fmt"
	"os"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type milvusPendingEntry struct {
	ID          string
	Model       string
	Query       string
	RequestBody string
}

type milvusBatchUpsertData struct {
	ids            []string
	requestIDs     []string
	models         []string
	queries        []string
	requestBodies  []string
	responseBodies []string
	embeddings     [][]float32
	timestamps     []int64
	embeddingDim   int
}

type milvusSingleUpsertData struct {
	id           string
	requestID    string
	model        string
	query        string
	requestBody  string
	responseBody string
	embedding    []float32
	timestamp    int64
	ttlSeconds   int64
	expiresAt    int64
}

func resolveMilvusConfig(options MilvusCacheOptions) (*config.MilvusConfig, error) {
	if options.Config != nil {
		return options.Config, nil
	}

	logging.Warnf("(Deprecated) MilvusCache: loading config from %s", options.ConfigPath)
	milvusConfig, err := loadMilvusConfig(options.ConfigPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load Milvus config: %w", err)
	}
	return milvusConfig, nil
}

func connectMilvusClient(milvusConfig *config.MilvusConfig) (client.Client, error) {
	connectionString := fmt.Sprintf("%s:%d", milvusConfig.Connection.Host, milvusConfig.Connection.Port)
	logging.Debugf("MilvusCache: connecting to Milvus at %s", connectionString)

	dialCtx := context.Background()
	if milvusConfig.Connection.Timeout > 0 {
		timeout := time.Duration(milvusConfig.Connection.Timeout) * time.Second
		var cancel context.CancelFunc
		dialCtx, cancel = context.WithTimeout(dialCtx, timeout)
		defer cancel()
		logging.Debugf("MilvusCache: connection timeout set to %s", timeout)
	}

	milvusClient, err := client.NewGrpcClient(dialCtx, connectionString)
	if err != nil {
		return nil, fmt.Errorf("failed to create Milvus client: %w", err)
	}
	return milvusClient, nil
}

func defaultMilvusEmbeddingModel(model string) string {
	if model == "" {
		return "bert"
	}
	return model
}

func closeMilvusClient(milvusClient client.Client, phase string) {
	if milvusClient == nil {
		return
	}
	if err := milvusClient.Close(); err != nil {
		logging.Warnf("MilvusCache: failed to close client after %s: %v", phase, err)
	}
}

func logParsedMilvusConfig(configPath string, milvusConfig *config.MilvusConfig) {
	logging.Debugf("MilvusConfig parsed from %s:\n", configPath)
	logging.Debugf("Collection.Name: %s\n", milvusConfig.Collection.Name)
	logging.Debugf("Collection.VectorField.Name: %s\n", milvusConfig.Collection.VectorField.Name)
	logging.Debugf("Collection.VectorField.Dimension: %d\n", milvusConfig.Collection.VectorField.Dimension)
	logging.Debugf("Collection.VectorField.MetricType: %s\n", milvusConfig.Collection.VectorField.MetricType)
	logging.Debugf("Collection.Index.Type: %s\n", milvusConfig.Collection.Index.Type)
	logging.Debugf("Development.AutoCreateCollection: %v\n", milvusConfig.Development.AutoCreateCollection)
	logging.Debugf("Development.DropCollectionOnStartup: %v\n", milvusConfig.Development.DropCollectionOnStartup)
}

func shouldForceMilvusDevelopmentSettings() bool {
	benchmarkMode := os.Getenv("SR_BENCHMARK_MODE")
	testMode := os.Getenv("SR_TEST_MODE")
	return benchmarkMode == "1" || benchmarkMode == "true" || testMode == "1" || testMode == "true"
}

func applyMilvusConfigEnvironmentOverrides(milvusConfig *config.MilvusConfig) {
	if !shouldForceMilvusDevelopmentSettings() {
		return
	}
	if milvusConfig.Development.AutoCreateCollection || milvusConfig.Development.DropCollectionOnStartup {
		return
	}

	logging.Warnf("Development settings parsed as false, forcing to true for benchmarks/tests\n")
	milvusConfig.Development.AutoCreateCollection = true
	milvusConfig.Development.DropCollectionOnStartup = true
}

func applyMilvusConfigDefaults(milvusConfig *config.MilvusConfig) {
	if milvusConfig.Collection.VectorField.Name == "" {
		logging.Warnf("VectorField.Name parsed as empty, setting to 'embedding'\n")
		milvusConfig.Collection.VectorField.Name = "embedding"
	}
	if milvusConfig.Collection.VectorField.MetricType == "" {
		logging.Warnf("VectorField.MetricType parsed as empty, setting to 'IP'\n")
		milvusConfig.Collection.VectorField.MetricType = "IP"
	}
	if milvusConfig.Collection.Index.Type == "" {
		logging.Warnf("Index.Type parsed as empty, setting to 'HNSW'\n")
		milvusConfig.Collection.Index.Type = "HNSW"
	}
	if milvusConfig.Collection.Index.Params.M == 0 {
		logging.Warnf("Index.Params.M parsed as 0, setting to 16\n")
		milvusConfig.Collection.Index.Params.M = 16
	}
	if milvusConfig.Collection.Index.Params.EfConstruction == 0 {
		logging.Warnf("Index.Params.EfConstruction parsed as 0, setting to 64\n")
		milvusConfig.Collection.Index.Params.EfConstruction = 64
	}
	if milvusConfig.Search.Params.Ef == 0 {
		logging.Warnf("Search.Params.Ef parsed as 0, setting to 64\n")
		milvusConfig.Search.Params.Ef = 64
	}
}

func resolveMilvusEntryID(id, model, query string, now time.Time) string {
	if id != "" {
		return id
	}
	return fmt.Sprintf("%x", md5.Sum(fmt.Appendf(nil, "%s_%s_%d", model, query, now.UnixNano())))
}

func resolveMilvusEffectiveTTL(ttlSeconds, defaultTTL int) int {
	if ttlSeconds == -1 {
		return defaultTTL
	}
	return ttlSeconds
}

func resolveMilvusExpiresAt(now time.Time, effectiveTTL int) int64 {
	if effectiveTTL > 0 {
		return now.Add(time.Duration(effectiveTTL) * time.Second).Unix()
	}
	return 0
}

func (c *MilvusCache) buildBatchUpsertData(entries []CacheEntry) (*milvusBatchUpsertData, error) {
	data := &milvusBatchUpsertData{
		ids:            make([]string, len(entries)),
		requestIDs:     make([]string, len(entries)),
		models:         make([]string, len(entries)),
		queries:        make([]string, len(entries)),
		requestBodies:  make([]string, len(entries)),
		responseBodies: make([]string, len(entries)),
		embeddings:     make([][]float32, len(entries)),
		timestamps:     make([]int64, len(entries)),
	}

	for i, entry := range entries {
		now := time.Now()
		embedding, err := c.getEmbedding(entry.Query)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embedding for entry %d: %w", i, err)
		}

		data.ids[i] = resolveMilvusEntryID("", entry.Model, entry.Query, now)
		data.requestIDs[i] = entry.RequestID
		data.models[i] = entry.Model
		data.queries[i] = entry.Query
		data.requestBodies[i] = string(entry.RequestBody)
		data.responseBodies[i] = string(entry.ResponseBody)
		data.embeddings[i] = embedding
		data.timestamps[i] = now.Unix()
	}

	if len(data.embeddings) > 0 {
		data.embeddingDim = len(data.embeddings[0])
	}
	return data, nil
}

func buildMilvusBatchColumns(data *milvusBatchUpsertData, vectorFieldName string) []entity.Column {
	return []entity.Column{
		entity.NewColumnVarChar("id", data.ids),
		entity.NewColumnVarChar("request_id", data.requestIDs),
		entity.NewColumnVarChar("model", data.models),
		entity.NewColumnVarChar("query", data.queries),
		entity.NewColumnVarChar("request_body", data.requestBodies),
		entity.NewColumnVarChar("response_body", data.responseBodies),
		entity.NewColumnFloatVector(vectorFieldName, data.embeddingDim, data.embeddings),
		entity.NewColumnInt64("timestamp", data.timestamps),
	}
}

func (c *MilvusCache) buildSingleUpsertData(id, requestID, model, query string, requestBody, responseBody []byte, ttlSeconds int) (*milvusSingleUpsertData, error) {
	effectiveTTL := resolveMilvusEffectiveTTL(ttlSeconds, c.ttlSeconds)
	embedding, err := c.getEmbedding(query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	now := time.Now()
	return &milvusSingleUpsertData{
		id:           resolveMilvusEntryID(id, model, query, now),
		requestID:    requestID,
		model:        model,
		query:        query,
		requestBody:  string(requestBody),
		responseBody: string(responseBody),
		embedding:    embedding,
		timestamp:    now.Unix(),
		ttlSeconds:   int64(effectiveTTL),
		expiresAt:    resolveMilvusExpiresAt(now, effectiveTTL),
	}, nil
}

func buildMilvusSingleColumns(data *milvusSingleUpsertData, vectorFieldName string) []entity.Column {
	return []entity.Column{
		entity.NewColumnVarChar("id", []string{data.id}),
		entity.NewColumnVarChar("request_id", []string{data.requestID}),
		entity.NewColumnVarChar("model", []string{data.model}),
		entity.NewColumnVarChar("query", []string{data.query}),
		entity.NewColumnVarChar("request_body", []string{data.requestBody}),
		entity.NewColumnVarChar("response_body", []string{data.responseBody}),
		entity.NewColumnFloatVector(vectorFieldName, len(data.embedding), [][]float32{data.embedding}),
		entity.NewColumnInt64("timestamp", []int64{data.timestamp}),
		entity.NewColumnInt64("ttl_seconds", []int64{data.ttlSeconds}),
		entity.NewColumnInt64("expires_at", []int64{data.expiresAt}),
	}
}

func milvusColumnValueAt(col entity.Column, idx int) (string, bool) {
	valueColumn, ok := col.(*entity.ColumnVarChar)
	if !ok || valueColumn.Len() <= idx {
		return "", false
	}

	value, err := valueColumn.ValueByIdx(idx)
	if err != nil {
		return "", false
	}
	return value, true
}

func milvusFirstColumnValue(col entity.Column) (string, bool) {
	return milvusColumnValueAt(col, 0)
}

func isLikelyMilvusPrimaryKey(value string) bool {
	return len(value) == 32 && isHexString(value)
}

func extractPendingEntry(columns []entity.Column) (milvusPendingEntry, error) {
	if len(columns) < 3 {
		return milvusPendingEntry{}, fmt.Errorf("incomplete query result: expected 3+ columns, got %d", len(columns))
	}

	entry := milvusPendingEntry{}
	dataValues := make([]string, 0, 3)
	for _, column := range columns {
		value, ok := milvusFirstColumnValue(column)
		if !ok {
			continue
		}
		if entry.ID == "" && isLikelyMilvusPrimaryKey(value) {
			entry.ID = value
			continue
		}
		dataValues = append(dataValues, value)
	}

	if entry.ID == "" || len(dataValues) < 3 || dataValues[0] == "" || dataValues[1] == "" {
		return milvusPendingEntry{}, fmt.Errorf("failed to extract required fields from query result")
	}

	entry.Model = dataValues[0]
	entry.Query = dataValues[1]
	entry.RequestBody = dataValues[2]
	return entry, nil
}

func extractMilvusResponseBody(columns []entity.Column, rowIndex int) ([]byte, error) {
	for _, column := range columns {
		value, ok := milvusColumnValueAt(column, rowIndex)
		if !ok || value == "" || isLikelyMilvusPrimaryKey(value) {
			continue
		}
		return []byte(value), nil
	}
	return nil, fmt.Errorf("response_body is empty")
}

func extractSearchDocumentContent(fields []entity.Column, rowIndex int) (string, bool) {
	for _, field := range fields {
		value, ok := milvusColumnValueAt(field, rowIndex)
		if !ok || value == "" || isLikelyMilvusPrimaryKey(value) {
			continue
		}
		return value, true
	}
	return "", false
}

func (c *MilvusCache) loadPendingEntry(requestID string) (milvusPendingEntry, error) {
	ctx := context.Background()
	queryExpr := fmt.Sprintf("request_id == \"%s\" && response_body == \"\"", requestID)
	logging.Debugf("MilvusCache.UpdateWithResponse: searching for pending entry with expr: %s", queryExpr)

	results, err := c.client.Query(ctx, c.collectionName, []string{}, queryExpr, []string{"model", "query", "request_body"})
	if err != nil {
		return milvusPendingEntry{}, fmt.Errorf("failed to query pending entry: %w", err)
	}
	if len(results) == 0 {
		return milvusPendingEntry{}, fmt.Errorf("no pending entry found")
	}
	return extractPendingEntry(results)
}

func (c *MilvusCache) collectionRowCount() int {
	if !c.enabled || c.client == nil {
		return 0
	}

	stats, err := c.client.GetCollectionStatistics(context.Background(), c.collectionName)
	if err != nil {
		logging.Debugf("MilvusCache.GetStats: failed to get collection stats: %v", err)
		return 0
	}

	entityCount, ok := stats["row_count"]
	if !ok {
		return 0
	}

	totalEntries := 0
	_, _ = fmt.Sscanf(entityCount, "%d", &totalEntries)
	if totalEntries > 0 {
		logging.Debugf("MilvusCache.GetStats: collection '%s' contains %d entries", c.collectionName, totalEntries)
	}
	return totalEntries
}
