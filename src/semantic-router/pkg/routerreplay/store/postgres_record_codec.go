package store

import (
	"database/sql"
	"encoding/json"
)

func nullableIntArg(value *int) interface{} {
	if value == nil {
		return nil
	}
	return *value
}

func nullableFloat64Arg(value *float64) interface{} {
	if value == nil {
		return nil
	}
	return *value
}

func nullableStringArg(value *string) interface{} {
	if value == nil {
		return nil
	}
	return *value
}

// emptyStringSQL maps "" to SQL NULL for optional VARCHAR columns.
func emptyStringSQL(s string) interface{} {
	if s == "" {
		return nil
	}
	return s
}

func assignUsageCostFields(
	record *Record,
	promptTokens sql.NullInt64,
	completionTokens sql.NullInt64,
	totalTokens sql.NullInt64,
	actualCost sql.NullFloat64,
	baselineCost sql.NullFloat64,
	costSavings sql.NullFloat64,
	currency sql.NullString,
	baselineModel sql.NullString,
) {
	if promptTokens.Valid {
		value := int(promptTokens.Int64)
		record.PromptTokens = &value
	}
	if completionTokens.Valid {
		value := int(completionTokens.Int64)
		record.CompletionTokens = &value
	}
	if totalTokens.Valid {
		value := int(totalTokens.Int64)
		record.TotalTokens = &value
	}
	if actualCost.Valid {
		value := actualCost.Float64
		record.ActualCost = &value
	}
	if baselineCost.Valid {
		value := baselineCost.Float64
		record.BaselineCost = &value
	}
	if costSavings.Valid {
		value := costSavings.Float64
		record.CostSavings = &value
	}
	if currency.Valid {
		value := currency.String
		record.Currency = &value
	}
	if baselineModel.Valid {
		value := baselineModel.String
		record.BaselineModel = &value
	}
}

func unmarshalReplayOptionalJSON(data []byte, target interface{}) error {
	if len(data) == 0 || string(data) == "null" {
		return nil
	}
	return json.Unmarshal(data, target)
}

func marshalReplayOptionalJSON(value interface{}) ([]byte, error) {
	if value == nil {
		return nil, nil
	}
	return json.Marshal(value)
}
