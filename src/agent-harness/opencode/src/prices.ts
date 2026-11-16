import { isRecord } from "./settings.ts";
// The host computes its estimate. An absent price table must never look free.
export function knownPrices(value: unknown): boolean {
  const rate = (input: unknown) =>
    typeof input === "number" && Number.isFinite(input) && input >= 0;
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (price) =>
        isRecord(price) &&
        rate(price.input) &&
        rate(price.output) &&
        (price.cache === undefined ||
          (isRecord(price.cache) &&
            rate(price.cache.read) &&
            rate(price.cache.write))),
    )
  );
}
