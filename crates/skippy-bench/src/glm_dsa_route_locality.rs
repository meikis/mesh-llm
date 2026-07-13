use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;

use crate::cli::GlmDsaRouteLocalityArgs;

const TRACE_MARKER: &str = "glm_dsa_tensor_trace ";
const ROUTE_TENSOR_NAME: &str = "ffn_moe_topk";

#[derive(Debug, Serialize)]
struct RouteLocalityReport {
    logs: Vec<LogRouteLocalitySummary>,
    combined: RouteLocalitySummary,
    cache_simulations: Vec<CacheSimulationSummary>,
}

#[derive(Debug, Serialize)]
struct LogRouteLocalitySummary {
    path: PathBuf,
    parsed_route_records: usize,
    ignored_non_decode_records: usize,
    ignored_incomplete_records: usize,
    summary: RouteLocalitySummary,
    cache_simulations: Vec<CacheSimulationSummary>,
    layers: Vec<LayerRouteLocalitySummary>,
}

#[derive(Clone, Debug, Serialize)]
struct CacheSimulationSummary {
    capacity_experts_per_layer: usize,
    layers: usize,
    transitions: usize,
    expert_lookups: usize,
    cache_hits: usize,
    cache_misses: usize,
    hit_fraction: Option<f64>,
    mean_misses_per_transition: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct RouteLocalitySummary {
    layers: usize,
    decode_records: usize,
    transitions: usize,
    compared_slots: usize,
    retained_expert_slots: usize,
    mean_expert_overlap: Option<f64>,
    mean_jaccard: Option<f64>,
    mean_rank_retention: Option<f64>,
    exact_set_reuse_fraction: Option<f64>,
    exact_order_reuse_fraction: Option<f64>,
    mean_new_experts_per_transition: Option<f64>,
    mean_unique_experts_per_layer: Option<f64>,
}

#[derive(Debug, Serialize)]
struct LayerRouteLocalitySummary {
    stage: i32,
    layer: u32,
    top_k: usize,
    decode_records: usize,
    transitions: usize,
    retained_expert_slots: usize,
    mean_expert_overlap: Option<f64>,
    mean_jaccard: Option<f64>,
    mean_rank_retention: Option<f64>,
    exact_set_reuse_fraction: Option<f64>,
    exact_order_reuse_fraction: Option<f64>,
    mean_new_experts_per_transition: Option<f64>,
    unique_experts: usize,
}

#[derive(Clone, Debug)]
struct RouteRecord {
    stage: i32,
    layer: u32,
    values: Vec<i32>,
}

#[derive(Clone, Debug, Default)]
struct LocalityAccumulator {
    layers: usize,
    decode_records: usize,
    transitions: usize,
    compared_slots: usize,
    retained_expert_slots: usize,
    rank_retained_slots: usize,
    jaccard_sum: f64,
    exact_set_reuses: usize,
    exact_order_reuses: usize,
    unique_experts: usize,
}

#[derive(Clone, Debug, Default)]
struct CacheAccumulator {
    capacity: usize,
    layers: usize,
    transitions: usize,
    lookups: usize,
    hits: usize,
}

pub fn glm_dsa_route_locality(args: GlmDsaRouteLocalityArgs) -> Result<()> {
    let output = args.output.clone();
    let report = build_report(&args)?;
    ensure!(
        report.combined.transitions >= args.min_transitions,
        "route-locality report contains {} transitions, fewer than required {}",
        report.combined.transitions,
        args.min_transitions
    );
    if let Some(required) = args.require_mean_overlap {
        ensure!(
            (0.0..=1.0).contains(&required),
            "--require-mean-overlap must be between 0 and 1"
        );
        let measured = report
            .combined
            .mean_expert_overlap
            .context("route-locality report has no mean overlap")?;
        ensure!(
            measured >= required,
            "mean route overlap {measured:.6} is below required {required:.6}"
        );
    }

    let encoded = serde_json::to_vec_pretty(&report)?;
    if let Some(path) = output {
        fs::write(&path, &encoded).with_context(|| format!("write {}", path.display()))?;
    }
    println!("{}", String::from_utf8(encoded)?);
    Ok(())
}

fn build_report(args: &GlmDsaRouteLocalityArgs) -> Result<RouteLocalityReport> {
    let cache_capacities = parse_cache_capacities(&args.cache_capacities)?;
    let mut logs = Vec::with_capacity(args.log.len());
    let mut combined = LocalityAccumulator::default();
    let mut combined_caches = cache_capacities
        .iter()
        .copied()
        .map(|capacity| (capacity, CacheAccumulator::new(capacity)))
        .collect::<BTreeMap<_, _>>();
    for path in &args.log {
        let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let text = select_log_window(path, &text, args)?;
        let summary = summarize_log(path.clone(), &text, &cache_capacities)?;
        combined.add_summary(&summary.summary);
        for cache in &summary.cache_simulations {
            combined_caches
                .get_mut(&cache.capacity_experts_per_layer)
                .expect("created every requested cache capacity")
                .add_summary(cache);
        }
        logs.push(summary);
    }
    Ok(RouteLocalityReport {
        logs,
        combined: combined.finish(),
        cache_simulations: combined_caches
            .into_values()
            .map(CacheAccumulator::finish)
            .collect(),
    })
}

fn summarize_log(
    path: PathBuf,
    text: &str,
    cache_capacities: &[usize],
) -> Result<LogRouteLocalitySummary> {
    let mut parsed_route_records = 0;
    let mut ignored_non_decode_records = 0;
    let mut ignored_incomplete_records = 0;
    let mut by_layer = BTreeMap::<(i32, u32), Vec<Vec<i32>>>::new();

    for line in text.lines().filter(|line| line.contains(TRACE_MARKER)) {
        let Some(parsed) = parse_trace_line(line)? else {
            ignored_incomplete_records += 1;
            continue;
        };
        parsed_route_records += 1;
        match parsed {
            ParsedTrace::NonDecode => ignored_non_decode_records += 1,
            ParsedTrace::Decode(record) => {
                by_layer
                    .entry((record.stage, record.layer))
                    .or_default()
                    .push(record.values);
            }
        }
    }
    ensure!(
        parsed_route_records > 0,
        "{} contains no {ROUTE_TENSOR_NAME} tensor traces",
        path.display()
    );

    let layers = by_layer
        .iter()
        .map(|(&(stage, layer), records)| summarize_layer(stage, layer, records))
        .collect::<Result<Vec<_>>>()?;
    let cache_simulations = cache_capacities
        .iter()
        .map(|&capacity| {
            let mut accumulator = CacheAccumulator::new(capacity);
            for records in by_layer.values() {
                accumulator.add_layer(records);
            }
            accumulator.finish()
        })
        .collect();
    let mut summary = LocalityAccumulator::default();
    for layer in &layers {
        summary.add_layer(layer);
    }

    Ok(LogRouteLocalitySummary {
        path,
        parsed_route_records,
        ignored_non_decode_records,
        ignored_incomplete_records,
        summary: summary.finish(),
        cache_simulations,
        layers,
    })
}

enum ParsedTrace {
    NonDecode,
    Decode(RouteRecord),
}

fn parse_trace_line(line: &str) -> Result<Option<ParsedTrace>> {
    let Some(start) = line.find(TRACE_MARKER) else {
        return Ok(None);
    };
    let fields = line[start + TRACE_MARKER.len()..]
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    let Some(name) = fields.get("name") else {
        return Ok(None);
    };
    if !name.starts_with(ROUTE_TENSOR_NAME) {
        return Ok(None);
    }
    let tokens = parse_required::<usize>(&fields, "tokens")?;
    if tokens != 1 {
        return Ok(Some(ParsedTrace::NonDecode));
    }
    ensure!(
        fields.get("type") == Some(&"i32"),
        "route tensor {name} is not i32"
    );
    let stage = parse_required::<i32>(&fields, "stage")?;
    let layer = name
        .rsplit_once('-')
        .and_then(|(_, suffix)| suffix.parse::<u32>().ok())
        .with_context(|| format!("route tensor name has no layer suffix: {name}"))?;
    let values = parse_i32_list(
        fields
            .get("values")
            .with_context(|| format!("route tensor {name} has no values"))?,
    )?;
    let shape = parse_i64_list(
        fields
            .get("ne")
            .with_context(|| format!("route tensor {name} has no shape"))?,
    )?;
    ensure!(
        shape.len() >= 2 && shape[1] == 1,
        "decode route tensor {name} is not one row: {shape:?}"
    );
    ensure!(
        !values.is_empty() && values.len() == shape[0] as usize,
        "route tensor {name} captured {} values for top-k {}",
        values.len(),
        shape[0]
    );
    Ok(Some(ParsedTrace::Decode(RouteRecord {
        stage,
        layer,
        values,
    })))
}

fn parse_required<T>(fields: &BTreeMap<&str, &str>, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    fields
        .get(name)
        .with_context(|| format!("tensor trace has no {name}"))?
        .parse::<T>()
        .map_err(|error| anyhow::anyhow!("invalid {name}: {error}"))
}

fn parse_i32_list(value: &str) -> Result<Vec<i32>> {
    parse_list(value, "i32 route value")
}

fn parse_i64_list(value: &str) -> Result<Vec<i64>> {
    parse_list(value, "shape value")
}

fn parse_list<T>(value: &str, label: &str) -> Result<Vec<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let body = value
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .with_context(|| format!("invalid {label} list: {value}"))?;
    if body.is_empty() {
        return Ok(Vec::new());
    }
    body.split(',')
        .map(|item| {
            item.parse::<T>()
                .map_err(|error| anyhow::anyhow!("invalid {label} {item}: {error}"))
        })
        .collect()
}

fn summarize_layer(
    stage: i32,
    layer: u32,
    records: &[Vec<i32>],
) -> Result<LayerRouteLocalitySummary> {
    let top_k = records.first().map_or(0, Vec::len);
    ensure!(top_k > 0, "stage {stage} layer {layer} has no route values");
    ensure!(
        records.iter().all(|record| record.len() == top_k),
        "stage {stage} layer {layer} has inconsistent top-k widths"
    );

    let unique = records.iter().flatten().copied().collect::<BTreeSet<_>>();
    let mut accumulator = LocalityAccumulator {
        layers: 1,
        decode_records: records.len(),
        unique_experts: unique.len(),
        ..LocalityAccumulator::default()
    };
    for window in records.windows(2) {
        accumulator.add_transition(&window[0], &window[1]);
    }
    let summary = accumulator.finish();
    Ok(LayerRouteLocalitySummary {
        stage,
        layer,
        top_k,
        decode_records: summary.decode_records,
        transitions: summary.transitions,
        retained_expert_slots: summary.retained_expert_slots,
        mean_expert_overlap: summary.mean_expert_overlap,
        mean_jaccard: summary.mean_jaccard,
        mean_rank_retention: summary.mean_rank_retention,
        exact_set_reuse_fraction: summary.exact_set_reuse_fraction,
        exact_order_reuse_fraction: summary.exact_order_reuse_fraction,
        mean_new_experts_per_transition: summary.mean_new_experts_per_transition,
        unique_experts: unique.len(),
    })
}

impl LocalityAccumulator {
    fn add_transition(&mut self, previous: &[i32], current: &[i32]) {
        let previous_set = previous.iter().copied().collect::<BTreeSet<_>>();
        let current_set = current.iter().copied().collect::<BTreeSet<_>>();
        let retained = previous_set.intersection(&current_set).count();
        let union = previous_set.union(&current_set).count();
        self.transitions += 1;
        self.compared_slots += current.len();
        self.retained_expert_slots += retained;
        self.rank_retained_slots += previous
            .iter()
            .zip(current)
            .filter(|(left, right)| left == right)
            .count();
        self.jaccard_sum += retained as f64 / union as f64;
        self.exact_set_reuses += usize::from(previous_set == current_set);
        self.exact_order_reuses += usize::from(previous == current);
    }

    fn add_layer(&mut self, layer: &LayerRouteLocalitySummary) {
        self.layers += 1;
        self.decode_records += layer.decode_records;
        self.transitions += layer.transitions;
        self.compared_slots += layer.transitions * layer.top_k;
        self.retained_expert_slots += layer.retained_expert_slots;
        self.rank_retained_slots +=
            fraction_count(layer.mean_rank_retention, layer.transitions * layer.top_k);
        self.jaccard_sum += layer.mean_jaccard.unwrap_or(0.0) * layer.transitions as f64;
        self.exact_set_reuses += fraction_count(layer.exact_set_reuse_fraction, layer.transitions);
        self.exact_order_reuses +=
            fraction_count(layer.exact_order_reuse_fraction, layer.transitions);
        self.unique_experts += layer.unique_experts;
    }

    fn add_summary(&mut self, summary: &RouteLocalitySummary) {
        self.layers += summary.layers;
        self.decode_records += summary.decode_records;
        self.transitions += summary.transitions;
        self.compared_slots += summary.compared_slots;
        self.retained_expert_slots += summary.retained_expert_slots;
        self.rank_retained_slots +=
            fraction_count(summary.mean_rank_retention, summary.compared_slots);
        self.jaccard_sum += summary.mean_jaccard.unwrap_or(0.0) * summary.transitions as f64;
        self.exact_set_reuses +=
            fraction_count(summary.exact_set_reuse_fraction, summary.transitions);
        self.exact_order_reuses +=
            fraction_count(summary.exact_order_reuse_fraction, summary.transitions);
        self.unique_experts +=
            fraction_count(summary.mean_unique_experts_per_layer, summary.layers);
    }

    fn finish(self) -> RouteLocalitySummary {
        RouteLocalitySummary {
            layers: self.layers,
            decode_records: self.decode_records,
            transitions: self.transitions,
            compared_slots: self.compared_slots,
            retained_expert_slots: self.retained_expert_slots,
            mean_expert_overlap: ratio(self.retained_expert_slots, self.compared_slots),
            mean_jaccard: ratio_f64(self.jaccard_sum, self.transitions),
            mean_rank_retention: ratio(self.rank_retained_slots, self.compared_slots),
            exact_set_reuse_fraction: ratio(self.exact_set_reuses, self.transitions),
            exact_order_reuse_fraction: ratio(self.exact_order_reuses, self.transitions),
            mean_new_experts_per_transition: ratio_f64(
                (self.compared_slots - self.retained_expert_slots) as f64,
                self.transitions,
            ),
            mean_unique_experts_per_layer: ratio_f64(self.unique_experts as f64, self.layers),
        }
    }
}

impl CacheAccumulator {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            ..Self::default()
        }
    }

    fn add_layer(&mut self, records: &[Vec<i32>]) {
        self.layers += 1;
        let mut lru = VecDeque::<i32>::with_capacity(self.capacity);
        let mut record_iter = records.iter();
        if let Some(first) = record_iter.next() {
            for &expert in first {
                touch_lru(&mut lru, self.capacity, expert);
            }
        }
        for record in record_iter {
            self.transitions += 1;
            self.lookups += record.len();
            self.hits += record.iter().filter(|expert| lru.contains(expert)).count();
            for &expert in record {
                touch_lru(&mut lru, self.capacity, expert);
            }
        }
    }

    fn add_summary(&mut self, summary: &CacheSimulationSummary) {
        debug_assert_eq!(self.capacity, summary.capacity_experts_per_layer);
        self.layers += summary.layers;
        self.transitions += summary.transitions;
        self.lookups += summary.expert_lookups;
        self.hits += summary.cache_hits;
    }

    fn finish(self) -> CacheSimulationSummary {
        CacheSimulationSummary {
            capacity_experts_per_layer: self.capacity,
            layers: self.layers,
            transitions: self.transitions,
            expert_lookups: self.lookups,
            cache_hits: self.hits,
            cache_misses: self.lookups - self.hits,
            hit_fraction: ratio(self.hits, self.lookups),
            mean_misses_per_transition: ratio(self.lookups - self.hits, self.transitions),
        }
    }
}

fn touch_lru(lru: &mut VecDeque<i32>, capacity: usize, expert: i32) {
    if let Some(index) = lru.iter().position(|&cached| cached == expert) {
        lru.remove(index);
    } else if lru.len() == capacity {
        lru.pop_front();
    }
    lru.push_back(expert);
}

fn parse_cache_capacities(value: &str) -> Result<Vec<usize>> {
    let capacities = value
        .split(',')
        .map(|item| {
            item.parse::<usize>()
                .with_context(|| format!("invalid cache capacity {item:?}"))
        })
        .collect::<Result<BTreeSet<_>>>()?;
    ensure!(!capacities.is_empty(), "--cache-capacities cannot be empty");
    ensure!(
        capacities.iter().all(|&capacity| capacity > 0),
        "--cache-capacities must all be greater than zero"
    );
    Ok(capacities.into_iter().collect())
}

fn fraction_count(fraction: Option<f64>, total: usize) -> usize {
    (fraction.unwrap_or(0.0) * total as f64).round() as usize
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    ratio_f64(numerator as f64, denominator)
}

fn ratio_f64(numerator: f64, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator / denominator as f64)
}

fn select_log_window(path: &Path, text: &str, args: &GlmDsaRouteLocalityArgs) -> Result<String> {
    let lines = text.lines().collect::<Vec<_>>();
    let start = match &args.from_marker {
        Some(marker) => lines
            .iter()
            .position(|line| line.contains(marker))
            .with_context(|| {
                format!("{} does not contain from marker {marker:?}", path.display())
            })?,
        None => 0,
    };
    let end = match &args.until_marker {
        Some(marker) => lines[start + 1..]
            .iter()
            .position(|line| line.contains(marker))
            .map(|offset| start + 1 + offset)
            .with_context(|| {
                format!(
                    "{} does not contain until marker {marker:?}",
                    path.display()
                )
            })?,
        None => lines.len(),
    };
    if start >= end {
        bail!(
            "{} selected an empty or reversed log window",
            path.display()
        );
    }
    Ok(lines[start..end].join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(tokens: usize, layer: u32, values: &[i32]) -> String {
        format!(
            "skippy: glm_dsa_tensor_trace stage=1 tokens={tokens} op=routed_moe_route node=1 name=ffn_moe_topk-{layer} type=i32 ne=[{},1,1,1] nb=[4,32,32,32] contiguous=1 nbytes=32 values=[{}]",
            values.len(),
            values
                .iter()
                .map(i32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    #[test]
    fn summarizes_consecutive_decode_routes_without_prefill() {
        let text = [
            trace(32, 75, &[1, 2, 3, 4]),
            trace(1, 75, &[1, 2, 3, 4]),
            trace(1, 75, &[1, 2, 5, 6]),
            trace(1, 75, &[1, 2, 5, 6]),
        ]
        .join("\n");
        let report = summarize_log(PathBuf::from("fixture.log"), &text, &[4, 8]).unwrap();
        assert_eq!(report.ignored_non_decode_records, 1);
        assert_eq!(report.summary.decode_records, 3);
        assert_eq!(report.summary.transitions, 2);
        assert_eq!(report.summary.retained_expert_slots, 6);
        assert_eq!(report.summary.mean_expert_overlap, Some(0.75));
        assert_eq!(report.summary.mean_new_experts_per_transition, Some(1.0));
        assert_eq!(report.summary.exact_set_reuse_fraction, Some(0.5));
        assert_eq!(report.summary.exact_order_reuse_fraction, Some(0.5));
        assert_eq!(report.cache_simulations[0].cache_hits, 6);
        assert_eq!(report.cache_simulations[0].cache_misses, 2);
    }

    #[test]
    fn keeps_layers_separate() {
        let text = [
            trace(1, 74, &[1, 2]),
            trace(1, 75, &[7, 8]),
            trace(1, 74, &[1, 3]),
            trace(1, 75, &[7, 8]),
        ]
        .join("\n");
        let report = summarize_log(PathBuf::from("fixture.log"), &text, &[2, 4]).unwrap();
        assert_eq!(report.summary.layers, 2);
        assert_eq!(report.summary.transitions, 2);
        assert_eq!(report.summary.mean_expert_overlap, Some(0.75));
        assert_eq!(report.layers[0].unique_experts, 3);
        assert_eq!(report.layers[1].unique_experts, 2);
        assert_eq!(report.cache_simulations[0].cache_hits, 3);
        assert_eq!(report.cache_simulations[0].cache_misses, 1);
    }
}
