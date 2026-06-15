package workflowstore

import "database/sql"

// ListMCPServerJSON returns persisted MCP server config JSON blobs ordered by id.
func (s *Store) ListMCPServerJSON() ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	rows, err := s.db.Query(`SELECT json FROM mcp_server ORDER BY id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []string
	for rows.Next() {
		var json string
		if err := rows.Scan(&json); err != nil {
			return nil, err
		}
		out = append(out, json)
	}
	return out, rows.Err()
}

// PutMCPServerJSON upserts one MCP server configuration row.
func (s *Store) PutMCPServerJSON(id, json string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(
		`INSERT INTO mcp_server (id, json) VALUES (?, ?)
		 ON CONFLICT(id) DO UPDATE SET json = excluded.json`,
		id, json,
	)
	return err
}

// DeleteMCPServer removes one MCP server configuration row.
func (s *Store) DeleteMCPServer(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	res, err := s.db.Exec(`DELETE FROM mcp_server WHERE id = ?`, id)
	if err != nil {
		return err
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

// MCPServerCount returns the number of persisted MCP server configs.
func (s *Store) MCPServerCount() (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var n int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM mcp_server`).Scan(&n)
	return n, err
}
