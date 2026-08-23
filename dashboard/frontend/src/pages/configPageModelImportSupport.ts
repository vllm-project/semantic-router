export const normalizedPrefix = (value: string) => {
  const trimmed = value.trim()
  return trimmed && !trimmed.endsWith('/') ? `${trimmed}/` : trimmed
}
