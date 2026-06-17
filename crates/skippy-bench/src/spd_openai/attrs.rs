use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result};
use serde_json::Value;

pub(super) fn attrs_for<'a>(events: &'a [Value], event_name: &str) -> Vec<&'a Value> {
    events
        .iter()
        .filter(|event| event.get("event").and_then(Value::as_str) == Some(event_name))
        .filter_map(|event| event.get("attributes"))
        .collect()
}

pub(super) fn count_events_by_hf_index(
    stage_events: &[Vec<Value>],
    event_name: &str,
    hf_index_key: &str,
) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    for events in stage_events {
        for attrs in attrs_for(events, event_name) {
            let key = attr_string(attrs, hf_index_key)
                .or_else(|| attr_i64(attrs, hf_index_key).map(|value| value.to_string()))
                .unwrap_or_else(|| "unknown".to_string());
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

pub(super) fn count_events(stage_events: &[Vec<Value>], event_name: &str) -> usize {
    stage_events
        .iter()
        .map(|events| attrs_for(events, event_name).len())
        .sum()
}

pub(super) fn read_events(path: &Path) -> Result<Vec<Value>> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut events = Vec::new();
    for line in content.lines() {
        if !line.starts_with('{') {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<Value>(line) {
            events.push(value);
        }
    }
    Ok(events)
}

pub(super) fn attr_string(attrs: &Value, key: &str) -> Option<String> {
    attrs
        .get(key)
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

pub(super) fn attr_bool(attrs: &Value, key: &str) -> Option<bool> {
    attrs.get(key).and_then(Value::as_bool)
}

pub(super) fn attr_f64(attrs: &Value, key: &str) -> Option<f64> {
    attrs.get(key).and_then(Value::as_f64)
}

pub(super) fn attr_u64(attrs: &Value, key: &str) -> Option<u64> {
    attrs.get(key).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
    })
}

pub(super) fn attr_i64(attrs: &Value, key: &str) -> Option<i64> {
    attrs.get(key).and_then(Value::as_i64)
}

pub(super) fn attr_i64_array(attrs: &Value, key: &str) -> Vec<i64> {
    attrs
        .get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_i64)
        .collect()
}

pub(super) fn attr_f64_array(attrs: &Value, key: &str) -> Vec<f64> {
    attrs
        .get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_f64)
        .collect()
}

pub(super) fn attr_i64_array_map(attrs: &Value, key: &str) -> BTreeMap<String, Vec<i64>> {
    attrs
        .get(key)
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|object| object.iter())
        .map(|(key, value)| {
            let values = value
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_i64)
                .collect();
            (key.clone(), values)
        })
        .collect()
}

pub(super) fn attr_u64_array(attrs: &Value, key: &str) -> Vec<u64> {
    attrs
        .get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|value| {
            value
                .as_u64()
                .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
        })
        .collect()
}
