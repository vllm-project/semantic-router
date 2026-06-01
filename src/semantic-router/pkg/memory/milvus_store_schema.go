package memory

import (
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func memoryCollectionSchema(collectionName string, dimension int) *entity.Schema {
	return &entity.Schema{
		CollectionName: collectionName,
		Description:    "Agentic Memory storage for cross-session context",
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				TypeParams: map[string]string{"max_length": "64"},
			},
			{
				Name:           "user_id",
				DataType:       entity.FieldTypeVarChar,
				TypeParams:     map[string]string{"max_length": "256"},
				IsPartitionKey: true,
			},
			{
				Name:       "project_id",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "256"},
			},
			{
				Name:       "memory_type",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "32"},
			},
			{
				Name:       "content",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:       "source",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "256"},
			},
			{
				Name:       "metadata",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:     "embedding",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", dimension),
				},
			},
			{Name: "created_at", DataType: entity.FieldTypeInt64},
			{Name: "updated_at", DataType: entity.FieldTypeInt64},
			{Name: "access_count", DataType: entity.FieldTypeInt64},
			{Name: "importance", DataType: entity.FieldTypeFloat},
		},
	}
}
