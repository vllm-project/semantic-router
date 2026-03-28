import type { Listener } from "@/types/config";
import type { DSLFieldObject } from "@/types/dsl";

export type EditableListener = Listener & { timeout: string };

export const DEFAULT_LISTENER_PORT = 8899;

export function getObj(
  fields: DSLFieldObject,
  key: string,
): DSLFieldObject {
  const value = fields[key];
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as DSLFieldObject;
  }
  return {};
}

export function getBool(
  obj: DSLFieldObject,
  key: string,
  def = false,
): boolean {
  const value = obj[key];
  return typeof value === "boolean" ? value : def;
}

export function getStr(
  obj: DSLFieldObject,
  key: string,
  def = "",
): string {
  const value = obj[key];
  if (typeof value === "string") return value;
  if (typeof value === "number") return String(value);
  return def;
}

export function getNum(
  obj: DSLFieldObject,
  key: string,
  def = 0,
): number {
  const value = obj[key];
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    const parsed = parseFloat(value);
    if (!Number.isNaN(parsed)) return parsed;
  }
  return def;
}

export function getListeners(
  fields: DSLFieldObject,
  key: string,
): EditableListener[] {
  const value = fields[key];
  if (!Array.isArray(value)) return [];

  return value
    .map((entry, index) => {
      if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
        return null;
      }
      const obj = entry as DSLFieldObject;
      const port = getNum(obj, "port", DEFAULT_LISTENER_PORT + index);
      return {
        name: getStr(obj, "name", `http-${port}`),
        address: getStr(obj, "address", "0.0.0.0"),
        port,
        timeout: getStr(obj, "timeout", "300s"),
      };
    })
    .filter((listener): listener is EditableListener => listener !== null);
}
