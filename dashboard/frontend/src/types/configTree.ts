export type ConfigTreeScalar = string | number | boolean | null
export type ConfigTreeValue =
  | ConfigTreeScalar
  | ConfigTreeObject
  | ConfigTreeValue[]

export interface ConfigTreeObject {
  [key: string]: ConfigTreeValue | undefined
}

export const isConfigTreeObject = (
  value: ConfigTreeValue | unknown,
): value is ConfigTreeObject =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)

