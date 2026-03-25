package workflowstore

import (
	"database/sql"
)

// ReplaceOpenClawContainers replaces all container rows (full snapshot).
func (s *Store) ReplaceOpenClawContainers(jsonRows [][2]string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM openclaw_container`); err != nil {
		_ = tx.Rollback()
		return err
	}
	stmt, err := tx.Prepare(`INSERT INTO openclaw_container (name, json) VALUES (?, ?)`)
	if err != nil {
		_ = tx.Rollback()
		return err
	}
	for _, row := range jsonRows {
		if _, err := stmt.Exec(row[0], row[1]); err != nil {
			_ = stmt.Close()
			_ = tx.Rollback()
			return err
		}
	}
	_ = stmt.Close()
	return tx.Commit()
}

// ReplaceOpenClawTeams replaces all team rows.
func (s *Store) ReplaceOpenClawTeams(jsonRows [][2]string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM openclaw_team`); err != nil {
		_ = tx.Rollback()
		return err
	}
	stmt, err := tx.Prepare(`INSERT INTO openclaw_team (id, json) VALUES (?, ?)`)
	if err != nil {
		_ = tx.Rollback()
		return err
	}
	for _, row := range jsonRows {
		if _, err := stmt.Exec(row[0], row[1]); err != nil {
			_ = stmt.Close()
			_ = tx.Rollback()
			return err
		}
	}
	_ = stmt.Close()
	return tx.Commit()
}

// ReplaceOpenClawRooms replaces all room rows.
func (s *Store) ReplaceOpenClawRooms(jsonRows [][2]string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM openclaw_room`); err != nil {
		_ = tx.Rollback()
		return err
	}
	stmt, err := tx.Prepare(`INSERT INTO openclaw_room (id, json) VALUES (?, ?)`)
	if err != nil {
		_ = tx.Rollback()
		return err
	}
	for _, row := range jsonRows {
		if _, err := stmt.Exec(row[0], row[1]); err != nil {
			_ = stmt.Close()
			_ = tx.Rollback()
			return err
		}
	}
	_ = stmt.Close()
	return tx.Commit()
}

// ListOpenClawContainerJSON returns name -> json payloads ordered by name.
func (s *Store) ListOpenClawContainerJSON() ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rows, err := s.db.Query(`SELECT json FROM openclaw_container ORDER BY name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanJSONLines(rows)
}

// ListOpenClawTeamJSON returns team json payloads ordered by id.
func (s *Store) ListOpenClawTeamJSON() ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rows, err := s.db.Query(`SELECT json FROM openclaw_team ORDER BY id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanJSONLines(rows)
}

// ListOpenClawRoomJSON returns room json payloads ordered by id.
func (s *Store) ListOpenClawRoomJSON() ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rows, err := s.db.Query(`SELECT json FROM openclaw_room ORDER BY id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanJSONLines(rows)
}

func scanJSONLines(rows *sql.Rows) ([]string, error) {
	var out []string
	for rows.Next() {
		var j string
		if err := rows.Scan(&j); err != nil {
			return nil, err
		}
		out = append(out, j)
	}
	return out, rows.Err()
}

// ListOpenClawRoomMessages returns message JSON payloads for a room in order.
func (s *Store) ListOpenClawRoomMessages(roomID string) ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rows, err := s.db.Query(
		`SELECT json FROM openclaw_room_message WHERE room_id = ? ORDER BY seq ASC`, roomID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanJSONLines(rows)
}

// AppendOpenClawRoomMessage inserts one message (O(1) vs rewriting a JSON file).
func (s *Store) AppendOpenClawRoomMessage(roomID, messageID, messageJSON string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.db.Exec(
		`INSERT OR REPLACE INTO openclaw_room_message (room_id, message_id, json) VALUES (?, ?, ?)`,
		roomID, messageID, messageJSON)
	return err
}

// ReplaceOpenClawRoomMessages replaces all messages for a room (full snapshot).
func (s *Store) ReplaceOpenClawRoomMessages(roomID string, messageIDAndJSON [][2]string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM openclaw_room_message WHERE room_id = ?`, roomID); err != nil {
		_ = tx.Rollback()
		return err
	}
	stmt, err := tx.Prepare(`INSERT INTO openclaw_room_message (room_id, message_id, json) VALUES (?, ?, ?)`)
	if err != nil {
		_ = tx.Rollback()
		return err
	}
	for _, row := range messageIDAndJSON {
		if _, err := stmt.Exec(roomID, row[0], row[1]); err != nil {
			_ = stmt.Close()
			_ = tx.Rollback()
			return err
		}
	}
	_ = stmt.Close()
	return tx.Commit()
}

// DeleteOpenClawRoomMessages removes all messages for a room.
func (s *Store) DeleteOpenClawRoomMessages(roomID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.db.Exec(`DELETE FROM openclaw_room_message WHERE room_id = ?`, roomID)
	return err
}

// DeleteOpenClawMessagesForRooms removes messages for multiple rooms (e.g. team teardown).
func (s *Store) DeleteOpenClawMessagesForRooms(roomIDs []string) error {
	if len(roomIDs) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	stmt, err := tx.Prepare(`DELETE FROM openclaw_room_message WHERE room_id = ?`)
	if err != nil {
		_ = tx.Rollback()
		return err
	}
	for _, id := range roomIDs {
		if _, err := stmt.Exec(id); err != nil {
			_ = stmt.Close()
			_ = tx.Rollback()
			return err
		}
	}
	_ = stmt.Close()
	return tx.Commit()
}

// OpenClawEntityCounts returns row counts for health reporting.
func (s *Store) OpenClawEntityCounts() (containers, teams, rooms, messages int, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err = s.db.QueryRow(`SELECT COUNT(*) FROM openclaw_container`).Scan(&containers); err != nil {
		return
	}
	if err = s.db.QueryRow(`SELECT COUNT(*) FROM openclaw_team`).Scan(&teams); err != nil {
		return
	}
	if err = s.db.QueryRow(`SELECT COUNT(*) FROM openclaw_room`).Scan(&rooms); err != nil {
		return
	}
	if err = s.db.QueryRow(`SELECT COUNT(*) FROM openclaw_room_message`).Scan(&messages); err != nil {
		return
	}
	return
}
