const escapeRegex = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Apply a visual mutation only inside the selected Recipe declaration. */
export function mutateBuilderRecipeSource(
  source: string,
  recipeName: string | null,
  mutation: (programSource: string) => string,
): string {
  if (!recipeName) return source
  const header = new RegExp(
    `^RECIPE\\s+${escapeRegex(recipeName)}\\s*(?:\\([^)]*\\))?\\s*\\{`,
    'm',
  ).exec(source)
  if (!header) return source

  const openBrace = source.indexOf('{', header.index)
  let depth = 0
  let closeBrace = -1
  for (let index = openBrace; index < source.length; index += 1) {
    if (source[index] === '{') depth += 1
    if (source[index] === '}') {
      depth -= 1
      if (depth === 0) {
        closeBrace = index
        break
      }
    }
  }
  if (closeBrace < 0) return source

  const rawBody = source.slice(openBrace + 1, closeBrace)
  const bodyLines = rawBody
    .replace(/^\r?\n/, '')
    .replace(/\r?\n[ \t]*$/, '')
    .split(/\r?\n/)
  const indents = bodyLines
    .filter((line) => line.trim())
    .map((line) => line.match(/^[ \t]*/)?.[0].length ?? 0)
  const indentSize = indents.length > 0 ? Math.min(...indents) : 2
  const indent = ' '.repeat(indentSize || 2)
  const body = bodyLines
    .map((line) => (line.trim() ? line.slice(Math.min(indentSize, line.length)) : ''))
    .join('\n')
  const mutated = mutation(body).trim()
  const reindented = mutated
    .split('\n')
    .map((line) => (line ? `${indent}${line}` : ''))
    .join('\n')
  return `${source.slice(0, openBrace + 1)}\n${reindented}\n${source.slice(closeBrace)}`
}
