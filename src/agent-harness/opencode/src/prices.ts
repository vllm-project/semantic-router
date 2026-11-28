import { isRecord } from "./settings.ts";
import type { Price } from "./turns.ts";
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

// Each model's list price as OpenCode lists it, by `provider/model`; a model with no known price is left out.
export function pricesOf(
  models: readonly { providerID: string; id: string; cost?: unknown }[],
): Map<string, Price> {
  const prices = new Map<string, Price>();
  for (const model of models) {
    if (!knownPrices(model.cost)) continue;
    const price = (model.cost as Record<string, unknown>[])[0]!;
    const cache = isRecord(price.cache) ? price.cache : undefined;
    prices.set(`${model.providerID}/${model.id}`, {
      input: price.input as number,
      output: price.output as number,
      cacheRead: (cache?.read as number | undefined) ?? 0,
      cacheWrite: (cache?.write as number | undefined) ?? 0,
    });
  }
  return prices;
}
