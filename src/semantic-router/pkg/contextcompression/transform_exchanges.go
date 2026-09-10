package contextcompression

import "sort"

// labelToolExchanges groups every parallel result with its owning assistant.
// The fixed point also handles messages containing both results and new calls.
func (request *RequestIR) labelToolExchanges() {
	groups := make(map[string][]*MessageIR)
	owners := make(map[string]int)
	results := make(map[string]int)
	for _, message := range request.Messages {
		ids := make(map[string]bool)
		for _, id := range messageToolCallIDs(message) {
			ids[id] = true
			owners[id]++
		}
		for _, id := range messageResultIDs(message) {
			ids[id] = true
			results[id]++
		}
		for id := range ids {
			message.ExchangeIDs = append(message.ExchangeIDs, id)
			groups[id] = append(groups[id], message)
		}
	}
	for _, message := range request.Messages {
		sort.Strings(message.ExchangeIDs)
	}
	propagateExchangeProtection(groups, owners, results)
}

func propagateExchangeProtection(groups map[string][]*MessageIR, owners, results map[string]int) {
	changed := true
	for changed {
		changed = false
		for id, messages := range groups {
			protection := exchangeProtection(id, messages, owners, results)
			for _, message := range messages {
				if message.Protection|protection != message.Protection {
					message.Protection |= protection
					changed = true
				}
			}
		}
	}
}

func exchangeProtection(id string, messages []*MessageIR, owners, results map[string]int) Protection {
	var protection Protection
	if id == "" || owners[id] != 1 || results[id] != 1 {
		protection |= ProtectUnknown
	}
	for _, message := range messages {
		protection |= message.Protection
	}
	return protection
}

func messageResultIDs(message *MessageIR) []string {
	ids := make(map[string]bool)
	for _, id := range message.ToolResultIDs {
		ids[id] = true
	}
	content, _ := message.Raw["content"].([]interface{})
	for _, raw := range content {
		block, _ := raw.(map[string]interface{})
		if block["type"] == "tool_result" {
			id, _ := block["tool_use_id"].(string)
			ids[id] = true
		}
	}
	if message.ToolCallID != "" {
		ids[message.ToolCallID] = true
	}
	for _, block := range message.Blocks {
		if block.ToolCallID != "" {
			ids[block.ToolCallID] = true
		}
	}
	result := make([]string, 0, len(ids))
	for id := range ids {
		result = append(result, id)
	}
	return result
}
