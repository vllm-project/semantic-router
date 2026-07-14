import type { FieldConfig } from '../components/EditModal'
import { coreFieldsForKey } from './configPageRouterCoreFields'
import { featureFieldsForKey } from './configPageRouterFeatureFields'
import type { RouterSystemKey } from './configPageRouterDefaultsSupport'

export function fieldsForKey(key: RouterSystemKey): FieldConfig[] {
  return coreFieldsForKey(key) ?? featureFieldsForKey(key) ?? []
}
