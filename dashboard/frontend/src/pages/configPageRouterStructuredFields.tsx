import type { FieldConfig } from '../components/EditModal'
import ConfigPageRouterStructuredEditor from './ConfigPageRouterStructuredEditor'
import {
  ROUTER_STRUCTURED_FIELDS,
  type RouterStructuredFieldDefinition,
} from './configPageRouterStructuredSchema'
import type { RouterSystemKey } from './configPageRouterDefaultsSupport'

export function getRouterStructuredFieldDefinition(
  key: RouterSystemKey,
  name: string,
): RouterStructuredFieldDefinition {
  const definition = ROUTER_STRUCTURED_FIELDS[key]?.[name]
  if (!definition) {
    throw new Error(`Missing structured field schema for ${key}.${name}`)
  }
  return definition
}

export function routerStructuredField(key: RouterSystemKey, name: string): FieldConfig {
  const definition = getRouterStructuredFieldDefinition(key, name)
  return {
    name,
    label: definition.label,
    type: 'custom',
    description: definition.description,
    required: definition.schema.required,
    customRender: (value, onChange) => (
      <ConfigPageRouterStructuredEditor
        schema={definition.schema}
        value={value}
        onChange={onChange}
      />
    ),
  }
}
