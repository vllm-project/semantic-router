import React, { useState } from 'react'
import styles from './EndpointsEditor.module.css'

export interface Endpoint {
  name: string
  endpoint: string
  protocol: 'http' | 'https'
  weight: number
}

interface EndpointsEditorProps {
  endpoints: Endpoint[]
  onChange: (endpoints: Endpoint[]) => void
}

const EndpointsEditor: React.FC<EndpointsEditorProps> = ({ endpoints, onChange }) => {
  const [editingIndex, setEditingIndex] = useState<number | null>(null)

  const handleAdd = () => {
    const newEndpoint: Endpoint = {
      name: `endpoint-${endpoints.length + 1}`,
      endpoint: 'localhost:8000',
      protocol: 'http',
      weight: 1
    }
    onChange([...endpoints, newEndpoint])
    setEditingIndex(endpoints.length)
  }

  const handleUpdate = (index: number, field: keyof Endpoint, value: string | number) => {
    const updated = [...endpoints]
    updated[index] = { ...updated[index], [field]: value }
    onChange(updated)
  }

  const handleDelete = (index: number) => {
    onChange(endpoints.filter((_, i) => i !== index))
    if (editingIndex === index) {
      setEditingIndex(null)
    }
  }

  return (
    <div className={styles.container}>
      <div className={styles.list}>
        {endpoints.map((ep, index) => (
          <div key={index} className={styles.item}>
            <div className={styles.itemHeader}>
              <span className={styles.itemName}>{ep.name}</span>
              <div className={styles.itemActions}>
                <button
                  type="button"
                  className={styles.btnEdit}
                  onClick={() => setEditingIndex(editingIndex === index ? null : index)}
                >
                  {editingIndex === index ? 'Done' : 'Edit'}
                </button>
                <button
                  type="button"
                  className={styles.btnDelete}
                  onClick={() => handleDelete(index)}
                >
                  Delete
                </button>
              </div>
            </div>

            {editingIndex === index ? (
              <div className={styles.form}>
                <div className={styles.formRow}>
                  <label>Name</label>
                  <input
                    type="text"
                    value={ep.name}
                    onChange={(e) => handleUpdate(index, 'name', e.target.value)}
                    placeholder="endpoint-1"
                  />
                </div>
                <div className={styles.formRow}>
                  <label>Address</label>
                  <input
                    type="text"
                    value={ep.endpoint}
                    onChange={(e) => handleUpdate(index, 'endpoint', e.target.value)}
                    placeholder="localhost:8000"
                  />
                </div>
                <div className={styles.formRow}>
                  <label>Protocol</label>
                  <select
                    value={ep.protocol}
                    onChange={(e) => handleUpdate(index, 'protocol', e.target.value)}
                  >
                    <option value="http">HTTP</option>
                    <option value="https">HTTPS</option>
                  </select>
                </div>
                <div className={styles.formRow}>
                  <label>Weight</label>
                  <input
                    type="number"
                    min="1"
                    value={ep.weight}
                    onChange={(e) => handleUpdate(index, 'weight', parseInt(e.target.value) || 1)}
                    placeholder="1"
                  />
                </div>
              </div>
            ) : (
              <div className={styles.itemDetails}>
                <span className={styles.detail}>{ep.endpoint}</span>
                <span className={styles.detail}>
                  <span className={ep.protocol === 'https' ? styles.https : styles.http}>
                    {ep.protocol.toUpperCase()}
                  </span>
                </span>
                <span className={styles.detail}>Weight: {ep.weight}</span>
              </div>
            )}
          </div>
        ))}
      </div>

      <button type="button" className={styles.btnAdd} onClick={handleAdd}>
        + Add Endpoint
      </button>

      {endpoints.length === 0 && (
        <div className={styles.empty}>
          No endpoints configured. Click "Add Endpoint" to create one.
        </div>
      )}
    </div>
  )
}

export default EndpointsEditor

