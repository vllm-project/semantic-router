package accesscontrol

import (
	"context"

	"github.com/jackc/pgx/v5"
)

func replaceGroupBindings(ctx context.Context, tx pgx.Tx, subjectType, subjectID string, groupIDs []string) error {
	if _, err := tx.Exec(ctx, `DELETE FROM access_group_bindings WHERE subject_type=$1 AND subject_id=$2`, subjectType, subjectID); err != nil {
		return err
	}
	for _, groupID := range uniqueStrings(groupIDs) {
		result, err := tx.Exec(ctx, `
INSERT INTO access_group_bindings(group_id,subject_type,subject_id)
SELECT id,$2,$3 FROM access_groups WHERE id=$1`, groupID, subjectType, subjectID)
		if err != nil {
			return err
		}
		if result.RowsAffected() == 0 {
			return validationErrorf("access group %s does not exist", groupID)
		}
	}
	return nil
}
