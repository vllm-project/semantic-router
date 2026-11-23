// The router settings page edits the `global` override block of config.yaml as
// a plain JSON tree. Updates share structure, so an unchanged subtree keeps its
// identity and memoized sections skip re-rendering.

export type JsonRecord = Record<string, unknown>
export type SettingPath = readonly string[]

export function isRecord(value: unknown): value is JsonRecord {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

export function cloneJson<T>(value: T): T {
  return value === undefined ? value : (JSON.parse(JSON.stringify(value)) as T)
}

export function deepEqual(left: unknown, right: unknown): boolean {
  if (left === right) return true
  if (Array.isArray(left) && Array.isArray(right)) {
    return (
      left.length === right.length && left.every((item, index) => deepEqual(item, right[index]))
    )
  }
  if (isRecord(left) && isRecord(right)) {
    const leftKeys = Object.keys(left)
    const rightKeys = Object.keys(right)
    return (
      leftKeys.length === rightKeys.length &&
      leftKeys.every((key) => key in right && deepEqual(left[key], right[key]))
    )
  }
  return false
}

export function getAt(root: unknown, path: SettingPath): unknown {
  let current = root
  for (const segment of path) {
    if (Array.isArray(current)) {
      current = current[Number(segment)]
    } else if (isRecord(current)) {
      current = current[segment]
    } else {
      return undefined
    }
  }
  return current
}

export function hasAt(root: unknown, path: SettingPath): boolean {
  return getAt(root, path) !== undefined
}

/** A copy of root with value at path; intermediate objects are created as needed. */
export function setAt<T>(root: T, path: SettingPath, value: unknown): T {
  if (path.length === 0) return value as T
  const [head, ...rest] = path
  if (Array.isArray(root)) {
    const next = [...root]
    next[Number(head)] = setAt(root[Number(head)], rest, value)
    return next as T
  }
  const base: JsonRecord = isRecord(root) ? root : {}
  return { ...base, [head]: setAt(base[head], rest, value) } as T
}

/** A copy of root without path; objects the removal leaves empty go too. */
export function unsetAt<T>(root: T, path: SettingPath): T {
  if (path.length === 0 || !isRecord(root)) return root
  const [head, ...rest] = path
  if (!(head in root)) return root
  if (rest.length === 0) {
    const next: JsonRecord = { ...root }
    delete next[head]
    return next as T
  }
  const child = root[head]
  if (!isRecord(child)) return root
  const nextChild = unsetAt(child, rest)
  const next: JsonRecord = { ...root }
  if (Object.keys(nextChild).length === 0) delete next[head]
  else next[head] = nextChild
  return next as T
}

/**
 * The RFC 7396 merge patch that turns saved into draft: a removed key becomes
 * null, a changed mapping recurses, and anything else is replaced whole.
 */
export function mergePatchDiff(saved: unknown, draft: unknown): unknown {
  if (isRecord(saved) && isRecord(draft)) {
    const patch: JsonRecord = {}
    for (const key of new Set([...Object.keys(saved), ...Object.keys(draft)])) {
      if (!(key in draft) || draft[key] === undefined) {
        if (key in saved && saved[key] !== undefined) patch[key] = null
        continue
      }
      if (!(key in saved) || saved[key] === undefined) {
        patch[key] = cloneJson(draft[key])
        continue
      }
      const child = mergePatchDiff(saved[key], draft[key])
      if (child !== undefined) patch[key] = child
    }
    return Object.keys(patch).length > 0 ? patch : undefined
  }
  return deepEqual(saved, draft) ? undefined : cloneJson(draft)
}

/** Applies an RFC 7396 merge patch, returning a new tree. */
export function applyMergePatch(target: unknown, patch: unknown): unknown {
  if (!isRecord(patch)) return cloneJson(patch)
  const base: JsonRecord = isRecord(target) ? { ...target } : {}
  for (const [key, value] of Object.entries(patch)) {
    if (value === null) delete base[key]
    else base[key] = applyMergePatch(base[key], value)
  }
  return base
}

/** Leaf paths whose value differs between saved and draft; arrays count as one leaf. */
export function changedLeafPaths(
  saved: unknown,
  draft: unknown,
  prefix: string[] = [],
): string[][] {
  if (isRecord(saved) && isRecord(draft)) {
    return [...new Set([...Object.keys(saved), ...Object.keys(draft)])].flatMap((key) =>
      changedLeafPaths(saved[key], draft[key], [...prefix, key]),
    )
  }
  if (isRecord(saved) && draft === undefined) return leafPaths(saved, prefix)
  if (isRecord(draft) && saved === undefined) return leafPaths(draft, prefix)
  return deepEqual(saved, draft) ? [] : [prefix]
}

/** Every leaf path below value; an empty mapping counts as one leaf. */
export function leafPaths(value: unknown, prefix: string[] = []): string[][] {
  if (!isRecord(value)) return value === undefined ? [] : [prefix]
  const keys = Object.keys(value)
  if (keys.length === 0) return [prefix]
  return keys.flatMap((key) => leafPaths(value[key], [...prefix, key]))
}

export function pathKey(path: SettingPath): string {
  return path.join('.')
}

export function startsWithPath(path: SettingPath, prefix: SettingPath): boolean {
  return prefix.length <= path.length && prefix.every((segment, index) => path[index] === segment)
}

/**
 * Re-applies the edits made on top of previousSaved to nextSaved, so a refresh
 * of the saved configuration keeps unsaved work.
 */
export function rebaseDraft(
  previousSaved: unknown,
  draft: unknown,
  nextSaved: unknown,
): JsonRecord {
  const patch = mergePatchDiff(previousSaved, draft)
  const base = isRecord(nextSaved) ? cloneJson(nextSaved) : {}
  return patch === undefined ? base : (applyMergePatch(base, patch) as JsonRecord)
}
