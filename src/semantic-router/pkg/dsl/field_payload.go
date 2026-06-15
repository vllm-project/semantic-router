package dsl

import "gopkg.in/yaml.v2"

type dslFieldObject map[string]dslFieldValue

type dslFieldArray []dslFieldValue

type dslFieldValue struct {
	raw any
}

func dslFieldObjectFromValues(fields map[string]Value) dslFieldObject {
	result := make(dslFieldObject, len(fields))
	for key, value := range fields {
		result[key] = dslFieldValueFromValue(value)
	}
	return result
}

func dslFieldValueFromValue(value Value) dslFieldValue {
	switch typed := value.(type) {
	case StringValue:
		return dslFieldValue{raw: typed.V}
	case IntValue:
		return dslFieldValue{raw: typed.V}
	case FloatValue:
		return dslFieldValue{raw: typed.V}
	case BoolValue:
		return dslFieldValue{raw: typed.V}
	case ArrayValue:
		items := make(dslFieldArray, 0, len(typed.Items))
		for _, item := range typed.Items {
			items = append(items, dslFieldValueFromValue(item))
		}
		return dslFieldValue{raw: items}
	case ObjectValue:
		return dslFieldValue{raw: dslFieldObjectFromValues(typed.Fields)}
	default:
		return dslFieldValue{}
	}
}

func (o dslFieldObject) setString(key, value string) {
	o[key] = dslFieldValue{raw: value}
}

func (o dslFieldObject) marshalYAML() ([]byte, error) {
	return yaml.Marshal(o.asInterfaceMap())
}

func (o dslFieldObject) asInterfaceMap() map[string]any {
	result := make(map[string]any, len(o))
	for key, value := range o {
		result[key] = value.asInterface()
	}
	return result
}

func (a dslFieldArray) asInterfaceSlice() []any {
	result := make([]any, 0, len(a))
	for _, value := range a {
		result = append(result, value.asInterface())
	}
	return result
}

func (v dslFieldValue) asInterface() any {
	switch typed := v.raw.(type) {
	case dslFieldObject:
		return typed.asInterfaceMap()
	case dslFieldArray:
		return typed.asInterfaceSlice()
	default:
		return typed
	}
}
