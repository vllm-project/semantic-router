interface KnowledgeBaseActivation {
  activation_status?: string
  generated_runtime_hash?: string
}

// A KB document can be persisted before the router finishes preparing its
// replacement generation. Refresh the displayed live list only after that
// exact candidate becomes active.
export async function waitForKnowledgeBaseActivation(result: KnowledgeBaseActivation): Promise<void> {
  if (result.activation_status !== 'pending') return
  const target = result.generated_runtime_hash
  if (!target) throw new Error('Knowledge base saved; router activation is pending.')
  const controller = new AbortController()
  const deadline = setTimeout(() => controller.abort(), 20000)
  try {
    for (let attempt = 0; attempt < 20 && !controller.signal.aborted; attempt += 1) {
      const response = await fetch('/api/router/api/v1/config/hash', { signal: controller.signal })
      if (!response.ok) break
      const snapshot = await response.json() as { active_runtime_hash?: string }
      if (snapshot.active_runtime_hash === target) return
      await new Promise((resolve) => setTimeout(resolve, 1000))
    }
  } catch {
    // Persistence already succeeded; a polling failure is not a failed write.
  } finally {
    clearTimeout(deadline)
  }
  throw new Error('Knowledge base saved; router activation is still pending. The previous configuration remains active.')
}
