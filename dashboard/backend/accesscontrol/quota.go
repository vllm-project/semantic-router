package accesscontrol

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
)

const quotaHashTag = "{vllm-sr-access}"

var reserveScript = redis.NewScript(`
local count = tonumber(ARGV[1])
for n=0,count-1 do
  local keyBase = n*3
  local argBase = 2+n*5
  local rpm = tonumber(ARGV[argBase])
  local tpm = tonumber(ARGV[argBase+1])
  local daily = tonumber(ARGV[argBase+2])
  local tokens = tonumber(ARGV[argBase+3])
  local budget = ARGV[argBase+4]
  if rpm > 0 and tonumber(redis.call('GET',KEYS[keyBase+1]) or '0') + 1 > rpm then return {0,'rpm',budget} end
  if tpm > 0 and tonumber(redis.call('GET',KEYS[keyBase+2]) or '0') + tokens > tpm then return {0,'tpm',budget} end
  if daily > 0 and tonumber(redis.call('GET',KEYS[keyBase+3]) or '0') + tokens > daily then return {0,'daily_tokens',budget} end
end
for n=0,count-1 do
  local keyBase = n*3
  local argBase = 2+n*5
  local rpm = tonumber(ARGV[argBase])
  local tpm = tonumber(ARGV[argBase+1])
  local daily = tonumber(ARGV[argBase+2])
  local tokens = tonumber(ARGV[argBase+3])
  if rpm > 0 then redis.call('INCR',KEYS[keyBase+1]); redis.call('EXPIRE',KEYS[keyBase+1],70) end
  if tpm > 0 then redis.call('INCRBY',KEYS[keyBase+2],tokens); redis.call('EXPIRE',KEYS[keyBase+2],70) end
  if daily > 0 then redis.call('INCRBY',KEYS[keyBase+3],tokens); redis.call('EXPIRE',KEYS[keyBase+3],tonumber(ARGV[2+count*5])) end
end
return {1}
`)

var reconcileScript = redis.NewScript(`
local delta = tonumber(ARGV[1])
for i=1,#KEYS do
  if delta >= 0 then
    redis.call('INCRBY',KEYS[i],delta)
  else
    local next = tonumber(redis.call('GET',KEYS[i]) or '0') + delta
    if next < 0 then next = 0 end
    redis.call('SET',KEYS[i],next,'KEEPTTL')
  end
end
return 1
`)

type QuotaManager struct {
	client *redis.Client
}

type Reservation struct {
	Budgets         []Budget
	EstimatedTokens int64
	MinuteBucket    string
	DayBucket       string
}

type QuotaError struct {
	Dimension string
	BudgetID  string
}

func (e *QuotaError) Error() string {
	return fmt.Sprintf("%s quota exceeded for budget %s", e.Dimension, e.BudgetID)
}

func OpenQuotaManager(ctx context.Context, redisURL string) (*QuotaManager, error) {
	if redisURL == "" {
		return nil, errors.New("ACCESS_CONTROL_REDIS_URL is required")
	}
	options, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("parse access-control Redis URL: %w", err)
	}
	client := redis.NewClient(options)
	if err := client.Ping(ctx).Err(); err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("ping access-control Redis: %w", err)
	}
	return &QuotaManager{client: client}, nil
}

func (q *QuotaManager) Close() { _ = q.client.Close() }

func (q *QuotaManager) Reserve(ctx context.Context, budgets []Budget, estimatedTokens int64) (*Reservation, error) {
	if estimatedTokens < 1 {
		estimatedTokens = 1
	}
	now := time.Now().UTC()
	minute := now.Format("200601021504")
	day := now.Format("20060102")
	reservation := &Reservation{Budgets: budgets, EstimatedTokens: estimatedTokens, MinuteBucket: minute, DayBucket: day}
	if len(budgets) == 0 {
		return reservation, nil
	}
	keys := make([]string, 0, len(budgets)*3)
	args := make([]any, 0, 2+len(budgets)*5)
	args = append(args, len(budgets))
	for _, budget := range budgets {
		base := quotaHashTag + ":" + budget.ID
		keys = append(keys, base+":rpm:"+minute, base+":tpm:"+minute, base+":daily:"+day)
		args = append(args, budget.RPM, budget.TPM, budget.DailyTokens, estimatedTokens, budget.ID)
	}
	nextDay := now.Truncate(24 * time.Hour).Add(24*time.Hour + 5*time.Minute)
	args = append(args, max(int64(time.Until(nextDay).Seconds()), 300))
	result, err := reserveScript.Run(ctx, q.client, keys, args...).Slice()
	if err != nil {
		return nil, fmt.Errorf("global quota store unavailable: %w", err)
	}
	if len(result) == 0 || toInt64(result[0]) != 1 {
		dimension, budgetID := "quota", "unknown"
		if len(result) > 1 {
			dimension = fmt.Sprint(result[1])
		}
		if len(result) > 2 {
			budgetID = fmt.Sprint(result[2])
		}
		return nil, &QuotaError{Dimension: dimension, BudgetID: budgetID}
	}
	return reservation, nil
}

func (q *QuotaManager) Reconcile(ctx context.Context, reservation *Reservation, actualTokens int64) error {
	if reservation == nil || len(reservation.Budgets) == 0 {
		return nil
	}
	delta := actualTokens - reservation.EstimatedTokens
	if delta == 0 {
		return nil
	}
	keys := make([]string, 0, len(reservation.Budgets)*2)
	for _, budget := range reservation.Budgets {
		base := quotaHashTag + ":" + budget.ID
		if budget.TPM > 0 {
			keys = append(keys, base+":tpm:"+reservation.MinuteBucket)
		}
		if budget.DailyTokens > 0 {
			keys = append(keys, base+":daily:"+reservation.DayBucket)
		}
	}
	if len(keys) == 0 {
		return nil
	}
	return reconcileScript.Run(ctx, q.client, keys, delta).Err()
}

func toInt64(value any) int64 {
	switch typed := value.(type) {
	case int64:
		return typed
	case string:
		parsed, _ := strconv.ParseInt(typed, 10, 64)
		return parsed
	case []byte:
		parsed, _ := strconv.ParseInt(string(typed), 10, 64)
		return parsed
	default:
		return 0
	}
}
