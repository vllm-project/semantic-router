import { useId, useState } from 'react'

import { ObjectListEditorItem } from './ObjectListEditorItem'
import type { ObjectEditorField, ObjectListEditorProps } from './ObjectListEditorTypes'
import styles from './StructuredFieldEditors.module.css'
import { updateStructuredObjectField } from './structuredFieldEditorSupport'

export type {
  ObjectEditorField,
  ObjectEditorFieldType,
  ObjectListEditorProps,
} from './ObjectListEditorTypes'

export function ObjectListEditor<TItem extends object>({
  value,
  onChange,
  fields,
  createItem,
  addLabel = 'Add item',
  emptyLabel = 'No items configured.',
  itemLabel = (_item, index) => `Item ${index + 1}`,
  itemDescription,
  validateItem,
  minItems = 0,
  maxItems,
  disabled = false,
  readOnly = false,
}: ObjectListEditorProps<TItem>) {
  const editorId = useId()
  const [editingIndex, setEditingIndex] = useState<number | null>(value.length === 1 ? 0 : null)

  const updateItem = (index: number, field: ObjectEditorField<TItem>, nextValue: unknown) => {
    const nextItems = [...value]
    nextItems[index] = updateStructuredObjectField(
      nextItems[index],
      field.key,
      nextValue,
      field.preserveEmpty,
    )
    onChange(nextItems)
  }

  const removeItem = (index: number) => {
    if (value.length <= minItems) return
    onChange(value.filter((_, itemIndex) => itemIndex !== index))
    setEditingIndex((current) => {
      if (current === null) return null
      if (current === index) return null
      return current > index ? current - 1 : current
    })
  }

  const addItem = () => {
    const nextIndex = value.length
    onChange([...value, createItem(nextIndex)])
    setEditingIndex(nextIndex)
  }

  return (
    <div className={styles.editor}>
      <div className={styles.objectList}>
        {value.map((item, index) => (
          <ObjectListEditorItem
            key={index}
            editorId={editorId}
            item={item}
            index={index}
            fields={fields}
            label={itemLabel(item, index)}
            description={itemDescription?.(item, index)}
            errors={validateItem?.(item, index) ?? []}
            expanded={readOnly || editingIndex === index}
            removable={value.length > minItems}
            disabled={disabled}
            readOnly={readOnly}
            onToggle={() => setEditingIndex(editingIndex === index ? null : index)}
            onRemove={() => removeItem(index)}
            onChange={(field, nextValue) => updateItem(index, field, nextValue)}
          />
        ))}
      </div>

      {value.length === 0 ? <p className={styles.empty}>{emptyLabel}</p> : null}
      {!readOnly && (maxItems === undefined || value.length < maxItems) ? (
        <button type="button" className={styles.addButton} onClick={addItem} disabled={disabled}>
          <span aria-hidden="true">+</span>
          {addLabel}
        </button>
      ) : null}
    </div>
  )
}
