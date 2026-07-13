use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;

use crate::cli::GlmDsaRouteMassArgs;

const TRACE_MARKER: &str = "glm_dsa_tensor_trace ";
const ROUTE_WEIGHT_TENSOR_NAME: &str = "ffn_moe_route_weights";

#[derive(Debug, Serialize)]
struct RouteMassReport {
    logs: Vec<LogRouteMassSummary>,
    combined: RouteMassSummary,
}

#[derive(Debug, Serialize)]
struct LogRouteMassSummary {
    path: PathBuf,
    parsed_route_records: usize,
    ignored_non_decode_records: usize,
    summary: RouteMassSummary,
    layers: Vec<LayerRouteMassSummary>,
}

#[derive(Clone, Debug, Serialize)]
struct RouteMassSummary {
    layers: usize,
    decode_records: usize,
    top_k: usize,
    mean_total_route_weight: f64,
    mean_normalized_weight_by_rank: Vec<f64>,
    retained_mass_by_expert_count: Vec<RetainedMassSummary>,
    adaptive_policies: Vec<AdaptivePolicySummary>,
}

#[derive(Clone, Debug, Serialize)]
struct LayerRouteMassSummary {
    stage: i32,
    layer: u32,
    #[serde(flatten)]
    summary: RouteMassSummary,
}

#[derive(Clone, Debug, Serialize)]
struct RetainedMassSummary {
    active_experts: usize,
    mean: f64,
    p10: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    minimum: f64,
}

#[derive(Clone, Debug, Serialize)]
struct AdaptivePolicySummary {
    retained_mass_threshold: f64,
    mean_active_experts: f64,
    p50_active_experts: usize,
    p90_active_experts: usize,
    p99_active_experts: usize,
    max_active_experts: usize,
    mean_retained_mass: f64,
    expert_compute_fraction: f64,
    ideal_expert_compute_reduction: f64,
    active_expert_histogram: Vec<ExpertCountFrequency>,
}

#[derive(Clone, Debug, Serialize)]
struct ExpertCountFrequency {
    active_experts: usize,
    records: usize,
    fraction: f64,
}

#[derive(Clone, Debug)]
struct RouteMassRecord {
    stage: i32,
    layer: u32,
    total_weight: f64,
    normalized_weights: Vec<f64>,
}

pub fn glm_dsa_route_mass(args: GlmDsaRouteMassArgs) -> Result<()> {
    let output = args.output.clone();
    let thresholds = parse_thresholds(&args.thresholds)?;
    let mut logs = Vec::with_capacity(args.log.len());
    let mut combined_records = Vec::new();

    for path in &args.log {
        let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let selected = select_log_window(path, &text, &args)?;
        let (summary, records) = summarize_log(path.clone(), &selected, &thresholds)?;
        combined_records.extend(records);
        logs.push(summary);
    }

    ensure!(
        combined_records.len() >= args.min_decode_records,
        "route-mass report contains {} decode records, fewer than required {}",
        combined_records.len(),
        args.min_decode_records
    );
    let combined = summarize_records(&combined_records, &thresholds)?;
    let report = RouteMassReport { logs, combined };
    let encoded = serde_json::to_vec_pretty(&report)?;
    if let Some(path) = output {
        fs::write(&path, &encoded).with_context(|| format!("write {}", path.display()))?;
    }
    println!("{}", String::from_utf8(encoded)?);
    Ok(())
}

fn summarize_log(
    path: PathBuf,
    text: &str,
    thresholds: &[f64],
) -> Result<(LogRouteMassSummary, Vec<RouteMassRecord>)> {
    let mut parsed_route_records = 0;
    let mut ignored_non_decode_records = 0;
    let mut records = Vec::new();

    for line in text.lines().filter(|line| line.contains(TRACE_MARKER)) {
        let Some(parsed) = parse_trace_line(line)? else {
            continue;
        };
        parsed_route_records += 1;
        match parsed {
            ParsedTrace::NonDecode => ignored_non_decode_records += 1,
            ParsedTrace::Decode(record) => records.push(record),
        }
    }
    ensure!(
        parsed_route_records > 0,
        "{} contains no {ROUTE_WEIGHT_TENSOR_NAME} tensor traces",
        path.display()
    );
    ensure!(
        !records.is_empty(),
        "{} contains no decode {ROUTE_WEIGHT_TENSOR_NAME} tensor traces",
        path.display()
    );

    let mut by_layer = BTreeMap::<(i32, u32), Vec<RouteMassRecord>>::new();
    for record in &records {
        by_layer
            .entry((record.stage, record.layer))
            .or_default()
            .push(record.clone());
    }
    let layers = by_layer
        .into_iter()
        .map(|((stage, layer), records)| {
            Ok(LayerRouteMassSummary {
                stage,
                layer,
                summary: summarize_records(&records, thresholds)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let summary = summarize_records(&records, thresholds)?;

    Ok((
        LogRouteMassSummary {
            path,
            parsed_route_records,
            ignored_non_decode_records,
            summary,
            layers,
        },
        records,
    ))
}

enum ParsedTrace {
    NonDecode,
    Decode(RouteMassRecord),
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
    if !name.starts_with(ROUTE_WEIGHT_TENSOR_NAME) {
        return Ok(None);
    }
    let tokens = parse_required::<usize>(&fields, "tokens")?;
    if tokens != 1 {
        return Ok(Some(ParsedTrace::NonDecode));
    }
    ensure!(
        fields.get("type") == Some(&"f32"),
        "route-weight tensor {name} is not f32"
    );
    let stage = parse_required::<i32>(&fields, "stage")?;
    let layer = name
        .rsplit_once('-')
        .and_then(|(_, suffix)| suffix.parse::<u32>().ok())
        .with_context(|| format!("route-weight tensor name has no layer suffix: {name}"))?;
    let shape = parse_list::<i64>(
        fields
            .get("ne")
            .with_context(|| format!("route-weight tensor {name} has no shape"))?,
        "shape value",
    )?;
    ensure!(
        shape.len() >= 3 && shape[2] == 1,
        "decode route-weight tensor {name} is not one token: {shape:?}"
    );
    let values = parse_list::<f64>(
        fields
            .get("values")
            .with_context(|| format!("route-weight tensor {name} has no values"))?,
        "route weight",
    )?;
    let top_k = shape[0]
        .checked_mul(shape[1])
        .context("route-weight shape overflows")?;
    ensure!(
        top_k > 0 && values.len() == top_k as usize,
        "route-weight tensor {name} captured {} values for top-k {top_k}",
        values.len()
    );
    ensure!(
        values
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0),
        "route-weight tensor {name} contains a negative or non-finite weight"
    );
    let total_weight = values.iter().sum::<f64>();
    ensure!(
        total_weight > 0.0 && total_weight.is_finite(),
        "route-weight tensor {name} has invalid total weight {total_weight}"
    );
    let mut normalized_weights = values
        .into_iter()
        .map(|weight| weight / total_weight)
        .collect::<Vec<_>>();
    normalized_weights.sort_by(|left, right| right.total_cmp(left));

    Ok(Some(ParsedTrace::Decode(RouteMassRecord {
        stage,
        layer,
        total_weight,
        normalized_weights,
    })))
}

fn summarize_records(records: &[RouteMassRecord], thresholds: &[f64]) -> Result<RouteMassSummary> {
    ensure!(!records.is_empty(), "cannot summarize zero route records");
    let top_k = records[0].normalized_weights.len();
    ensure!(
        records
            .iter()
            .all(|record| record.normalized_weights.len() == top_k),
        "route records have inconsistent top-k widths"
    );
    let layers = records
        .iter()
        .map(|record| (record.stage, record.layer))
        .collect::<BTreeSet<_>>()
        .len();
    let mean_total_route_weight = records
        .iter()
        .map(|record| record.total_weight)
        .sum::<f64>()
        / records.len() as f64;
    let mean_normalized_weight_by_rank = (0..top_k)
        .map(|rank| {
            records
                .iter()
                .map(|record| record.normalized_weights[rank])
                .sum::<f64>()
                / records.len() as f64
        })
        .collect();
    let retained_mass_by_expert_count = (1..=top_k)
        .map(|active_experts| retained_mass_summary(records, active_experts))
        .collect();
    let adaptive_policies = thresholds
        .iter()
        .map(|&threshold| adaptive_policy_summary(records, threshold, top_k))
        .collect();

    Ok(RouteMassSummary {
        layers,
        decode_records: records.len(),
        top_k,
        mean_total_route_weight,
        mean_normalized_weight_by_rank,
        retained_mass_by_expert_count,
        adaptive_policies,
    })
}

fn retained_mass_summary(
    records: &[RouteMassRecord],
    active_experts: usize,
) -> RetainedMassSummary {
    let values = records
        .iter()
        .map(|record| {
            record.normalized_weights[..active_experts]
                .iter()
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    RetainedMassSummary {
        active_experts,
        mean: mean(&values),
        p10: percentile_f64(&values, 0.10),
        p50: percentile_f64(&values, 0.50),
        p90: percentile_f64(&values, 0.90),
        p99: percentile_f64(&values, 0.99),
        minimum: values.iter().copied().fold(f64::INFINITY, f64::min),
    }
}

fn adaptive_policy_summary(
    records: &[RouteMassRecord],
    threshold: f64,
    top_k: usize,
) -> AdaptivePolicySummary {
    let selections = records
        .iter()
        .map(|record| {
            let mut retained = 0.0;
            let mut active = top_k;
            for (index, weight) in record.normalized_weights.iter().enumerate() {
                retained += weight;
                if retained + f64::EPSILON >= threshold {
                    active = index + 1;
                    break;
                }
            }
            (active, retained)
        })
        .collect::<Vec<_>>();
    let counts = selections
        .iter()
        .map(|(active, _)| *active)
        .collect::<Vec<_>>();
    let mean_active_experts = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
    let mut histogram = BTreeMap::<usize, usize>::new();
    for &active in &counts {
        *histogram.entry(active).or_default() += 1;
    }

    AdaptivePolicySummary {
        retained_mass_threshold: threshold,
        mean_active_experts,
        p50_active_experts: percentile_usize(&counts, 0.50),
        p90_active_experts: percentile_usize(&counts, 0.90),
        p99_active_experts: percentile_usize(&counts, 0.99),
        max_active_experts: counts.iter().copied().max().unwrap_or(top_k),
        mean_retained_mass: selections.iter().map(|(_, mass)| *mass).sum::<f64>()
            / selections.len() as f64,
        expert_compute_fraction: mean_active_experts / top_k as f64,
        ideal_expert_compute_reduction: 1.0 - mean_active_experts / top_k as f64,
        active_expert_histogram: histogram
            .into_iter()
            .map(|(active_experts, count)| ExpertCountFrequency {
                active_experts,
                records: count,
                fraction: count as f64 / counts.len() as f64,
            })
            .collect(),
    }
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

fn parse_thresholds(value: &str) -> Result<Vec<f64>> {
    let mut thresholds = value
        .split(',')
        .map(|item| {
            item.parse::<f64>()
                .with_context(|| format!("invalid route-mass threshold {item:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!thresholds.is_empty(), "--thresholds cannot be empty");
    ensure!(
        thresholds
            .iter()
            .all(|threshold| threshold.is_finite() && *threshold > 0.0 && *threshold <= 1.0),
        "--thresholds must all be finite and in (0, 1]"
    );
    thresholds.sort_by(f64::total_cmp);
    thresholds.dedup_by(|left, right| left.total_cmp(right).is_eq());
    Ok(thresholds)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile_f64(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[percentile_index(sorted.len(), quantile)]
}

fn percentile_usize(values: &[usize], quantile: f64) -> usize {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[percentile_index(sorted.len(), quantile)]
}

fn percentile_index(len: usize, quantile: f64) -> usize {
    ((quantile * len as f64).ceil() as usize)
        .saturating_sub(1)
        .min(len - 1)
}

fn select_log_window(path: &Path, text: &str, args: &GlmDsaRouteMassArgs) -> Result<String> {
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

    fn trace(tokens: usize, stage: i32, layer: u32, weights: &[f64]) -> String {
        format!(
            "skippy: glm_dsa_tensor_trace stage={stage} tokens={tokens} op=routed_moe_route node=2 name=ffn_moe_route_weights-{layer} type=f32 ne=[1,{},1,1] nb=[4,4,32,32] contiguous=1 nbytes=32 values=[{}]",
            weights.len(),
            weights
                .iter()
                .map(f64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    #[test]
    fn summarizes_retained_mass_and_adaptive_counts() {
        let text = [
            trace(16, 0, 8, &[0.4, 0.3, 0.2, 0.1]),
            trace(1, 0, 8, &[0.4, 0.3, 0.2, 0.1]),
            trace(1, 0, 8, &[0.7, 0.1, 0.1, 0.1]),
        ]
        .join("\n");
        let (report, records) =
            summarize_log(PathBuf::from("fixture.log"), &text, &[0.75, 0.9]).unwrap();

        assert_eq!(report.ignored_non_decode_records, 1);
        assert_eq!(report.summary.decode_records, 2);
        assert_eq!(report.summary.top_k, 4);
        assert!((report.summary.retained_mass_by_expert_count[1].mean - 0.75).abs() < f64::EPSILON);
        assert_eq!(report.summary.adaptive_policies[0].mean_active_experts, 2.5);
        assert_eq!(report.summary.adaptive_policies[1].mean_active_experts, 3.0);
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn keeps_layer_and_stage_summaries_separate() {
        let text = [trace(1, 0, 8, &[0.8, 0.2]), trace(1, 1, 60, &[0.6, 0.4])].join("\n");
        let (report, _) = summarize_log(PathBuf::from("fixture.log"), &text, &[0.9]).unwrap();

        assert_eq!(report.summary.layers, 2);
        assert_eq!(report.layers.len(), 2);
        assert_eq!(report.layers[0].stage, 0);
        assert_eq!(report.layers[1].stage, 1);
    }
}
