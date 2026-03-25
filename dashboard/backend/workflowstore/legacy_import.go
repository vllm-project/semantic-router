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

	regPath := filepath.Join(dir, "containers.json")
	teamPath := filepath.Join(dir, "teams.json")
	roomPath := filepath.Join(dir, "rooms.json")

	var containers [][2]string
	if data, err := os.ReadFile(regPath); err == nil && len(data) > 0 {
		var raw []map[string]any
		if err := json.Unmarshal(data, &raw); err != nil {
			return err
		}
		for _, m := range raw {
			name, _ := m["name"].(string)
			if name == "" {
				continue
			}
			b, err := json.Marshal(m)
			if err != nil {
				return err
			}
			containers = append(containers, [2]string{name, string(b)})
		}
	}

	var teams [][2]string
	if data, err := os.ReadFile(teamPath); err == nil && len(data) > 0 {
		var raw []map[string]any
		if err := json.Unmarshal(data, &raw); err != nil {
			return err
		}
		for _, m := range raw {
			id, _ := m["id"].(string)
			if id == "" {
				continue
			}
			b, err := json.Marshal(m)
			if err != nil {
				return err
			}
			teams = append(teams, [2]string{id, string(b)})
		}
	}

	var rooms [][2]string
	if data, err := os.ReadFile(roomPath); err == nil && len(data) > 0 {
		var raw []map[string]any
		if err := json.Unmarshal(data, &raw); err != nil {
			return err
		}
		for _, m := range raw {
			id, _ := m["id"].(string)
			if id == "" {
				continue
			}
			b, err := json.Marshal(m)
			if err != nil {
				return err
			}
			rooms = append(rooms, [2]string{id, string(b)})
		}
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

	msgDir := filepath.Join(dir, "room-messages")
	entries, _ := os.ReadDir(msgDir)
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if filepath.Ext(e.Name()) != ".json" {
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

	return nil
}
