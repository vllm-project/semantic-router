/** Short labels follow first appearance; exact IDs remain available in the view. */
export function buildConversationLabels(
  records: ReadonlyArray<{ conversation_id?: string }>,
): Map<string, string> {
  const labels = new Map<string, string>()
  for (const record of records) {
    const id = record.conversation_id
    if (id && !labels.has(id)) labels.set(id, `Conversation ${labels.size + 1}`)
  }
  return labels
}
