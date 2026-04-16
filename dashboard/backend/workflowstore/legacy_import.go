package workflowstore

import (
	"encoding/json"
	"os"
	"path/filepath"
)

func (s *Store) maybeImportLegacyOpenClaw(dir string) error {
	var n int
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM openclaw_container`).Scan(&n); err != nil {
		return err
	}
	if n > 0 {
		return nil
	}

	containers, err := readLegacyJSONPairs(filepath.Join(dir, "containers.json"), "name")
	if err != nil {
		return err
	}
	teams, err := readLegacyJSONPairs(filepath.Join(dir, "teams.json"), "id")
	if err != nil {
		return err
	}
	rooms, err := readLegacyJSONPairs(filepath.Join(dir, "rooms.json"), "id")
	if err != nil {
		return err
	}

	if len(containers) > 0 {
		if err := s.ReplaceOpenClawContainers(containers); err != nil {
			return err
		}
	}
	if len(teams) > 0 {
		if err := s.ReplaceOpenClawTeams(teams); err != nil {
			return err
		}
	}
	if len(rooms) > 0 {
		if err := s.ReplaceOpenClawRooms(rooms); err != nil {
			return err
		}
	}

	s.importLegacyRoomMessages(filepath.Join(dir, "room-messages"))
	return nil
}

func readLegacyJSONPairs(path, keyField string) ([][2]string, error) {
	data, err := os.ReadFile(path)
	if err != nil || len(data) == 0 {
		return nil, nil
	}
	var raw []map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	var pairs [][2]string
	for _, m := range raw {
		key, _ := m[keyField].(string)
		if key == "" {
			continue
		}
		b, err := json.Marshal(m)
		if err != nil {
			return nil, err
		}
		pairs = append(pairs, [2]string{key, string(b)})
	}
	return pairs, nil
}

func (s *Store) importLegacyRoomMessages(msgDir string) {
	entries, _ := os.ReadDir(msgDir)
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		roomID := e.Name()[:len(e.Name())-len(".json")]
		data, err := os.ReadFile(filepath.Join(msgDir, e.Name()))
		if err != nil || len(data) == 0 {
			continue
		}
		var msgs []map[string]any
		if err := json.Unmarshal(data, &msgs); err != nil {
			continue
		}
		for _, m := range msgs {
			mid, _ := m["id"].(string)
			if mid == "" {
				continue
			}
			b, err := json.Marshal(m)
			if err != nil {
				continue
			}
			_ = s.AppendOpenClawRoomMessage(roomID, mid, string(b))
		}
	}
}
