use std::{collections::BTreeMap, fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::cli::GlmDsaOpReportArgs;

const PREFIX: &str = "skippy: glm_dsa_op_timing ";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Default, Serialize)]
struct OpBucket {
    nodes: u64,
    elapsed_us: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
struct PhaseSummary {
    records: usize,
    tokens: u64,
    total_us: u64,
    avg_total_us_per_record: Option<f64>,
    avg_total_us_per_token: Option<f64>,
    indexer_topk: OpBucket,
    sparse_mask: OpBucket,
    mla_attention: OpBucket,
    routed_moe: OpBucket,
    shared_expert: OpBucket,
}

#[derive(Debug, Clone, Serialize)]
struct LogSummary {
    path: PathBuf,
    records: usize,
    stage_records: BTreeMap<i32, BTreeMap<Phase, PhaseSummary>>,
}

#[derive(Debug, Serialize)]
struct GlmDsaOpReport {
    logs: Vec<LogSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TimingRecord {
    stage: i32,
    tokens: u64,
    total_us: u64,
    indexer_topk_nodes: u64,
    indexer_topk_us: u64,
    sparse_mask_nodes: u64,
    sparse_mask_us: u64,
    mla_attention_nodes: u64,
    mla_attention_us: u64,
    routed_moe_nodes: u64,
    routed_moe_us: u64,
    shared_expert_nodes: u64,
    shared_expert_us: u64,
}

pub fn glm_dsa_op_report(args: GlmDsaOpReportArgs) -> Result<()> {
    let output = args.output.clone();
    let report = build_report(&args)?;
    let encoded = serde_json::to_vec_pretty(&report)?;
    if let Some(path) = output {
        fs::write(&path, &encoded).with_context(|| format!("write {}", path.display()))?;
    }
    println!("{}", String::from_utf8(encoded)?);
    Ok(())
}

fn build_report(args: &GlmDsaOpReportArgs) -> Result<GlmDsaOpReport> {
    let mut logs = Vec::with_capacity(args.log.len());
    for path in &args.log {
        let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let records = parse_records(&text)
            .with_context(|| format!("parse GLM-DSA op timing records in {}", path.display()))?;
        if records.is_empty() {
            bail!("{} contains no GLM-DSA op timing records", path.display());
        }
        let records = match args.first_records {
            Some(limit) => records.into_iter().take(limit).collect::<Vec<_>>(),
            None => records,
        };
        logs.push(summarize_log(path.clone(), &records));
    }
    Ok(GlmDsaOpReport { logs })
}

fn parse_records(text: &str) -> Result<Vec<TimingRecord>> {
    text.lines()
        .filter_map(|line| line.find(PREFIX).map(|index| &line[index + PREFIX.len()..]))
        .map(parse_record)
        .collect()
}

fn parse_record(line: &str) -> Result<TimingRecord> {
    let fields = line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    Ok(TimingRecord {
        stage: parse_field(&fields, "stage")?,
        tokens: parse_field(&fields, "tokens")?,
        total_us: parse_field(&fields, "total_us")?,
        indexer_topk_nodes: parse_field(&fields, "indexer_topk_nodes")?,
        indexer_topk_us: parse_field(&fields, "indexer_topk_us")?,
        sparse_mask_nodes: parse_field(&fields, "sparse_mask_nodes")?,
        sparse_mask_us: parse_field(&fields, "sparse_mask_us")?,
        mla_attention_nodes: parse_field(&fields, "mla_attention_nodes")?,
        mla_attention_us: parse_field(&fields, "mla_attention_us")?,
        routed_moe_nodes: parse_field(&fields, "routed_moe_nodes")?,
        routed_moe_us: parse_field(&fields, "routed_moe_us")?,
        shared_expert_nodes: parse_field(&fields, "shared_expert_nodes")?,
        shared_expert_us: parse_field(&fields, "shared_expert_us")?,
    })
}

fn parse_field<T>(fields: &BTreeMap<&str, &str>, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    fields
        .get(name)
        .with_context(|| format!("missing {name}"))?
        .parse::<T>()
        .map_err(|error| anyhow::anyhow!("invalid {name}: {error}"))
}

fn summarize_log(path: PathBuf, records: &[TimingRecord]) -> LogSummary {
    let mut stage_records: BTreeMap<i32, BTreeMap<Phase, PhaseSummary>> = BTreeMap::new();
    for record in records {
        let phase = if record.tokens == 1 {
            Phase::Decode
        } else {
            Phase::Prefill
        };
        let summary = stage_records
            .entry(record.stage)
            .or_default()
            .entry(phase)
            .or_default();
        summary.records += 1;
        summary.tokens += record.tokens;
        summary.total_us += record.total_us;
        add_bucket(
            &mut summary.indexer_topk,
            record.indexer_topk_nodes,
            record.indexer_topk_us,
        );
        add_bucket(
            &mut summary.sparse_mask,
            record.sparse_mask_nodes,
            record.sparse_mask_us,
        );
        add_bucket(
            &mut summary.mla_attention,
            record.mla_attention_nodes,
            record.mla_attention_us,
        );
        add_bucket(
            &mut summary.routed_moe,
            record.routed_moe_nodes,
            record.routed_moe_us,
        );
        add_bucket(
            &mut summary.shared_expert,
            record.shared_expert_nodes,
            record.shared_expert_us,
        );
    }
    for phases in stage_records.values_mut() {
        for summary in phases.values_mut() {
            summary.avg_total_us_per_record = nonzero_div(summary.total_us, summary.records as u64);
            summary.avg_total_us_per_token = nonzero_div(summary.total_us, summary.tokens);
        }
    }
    LogSummary {
        path,
        records: records.len(),
        stage_records,
    }
}

fn add_bucket(bucket: &mut OpBucket, nodes: u64, elapsed_us: u64) {
    bucket.nodes += nodes;
    bucket.elapsed_us += elapsed_us;
}

fn nonzero_div(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

#[cfg(test)]
mod tests {
    use super::{Phase, parse_record, parse_records, summarize_log};

    const LINE: &str = "skippy: glm_dsa_op_timing stage=1 tokens=128 total_us=1475800 indexer_topk_nodes=275 indexer_topk_us=129065 sparse_mask_nodes=235 sparse_mask_us=114543 mla_attention_nodes=47 mla_attention_us=35234 routed_moe_nodes=47 routed_moe_us=379574 shared_expert_nodes=47 shared_expert_us=817384";

    #[test]
    fn parses_timing_record_with_prefix() {
        let records = parse_records(LINE).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].stage, 1);
        assert_eq!(records[0].tokens, 128);
        assert_eq!(records[0].indexer_topk_us, 129065);
        assert_eq!(records[0].shared_expert_nodes, 47);
    }

    #[test]
    fn rejects_missing_fields() {
        let error = parse_record("stage=0 tokens=1").unwrap_err().to_string();
        assert!(error.contains("missing total_us"));
    }

    #[test]
    fn summarizes_prefill_and_decode() {
        let text = format!(
            "{LINE}\n{}",
            LINE.replace("tokens=128", "tokens=1")
                .replace("total_us=1475800", "total_us=200")
        );
        let records = parse_records(&text).unwrap();
        let summary = summarize_log("stage1.log".into(), &records);
        let stages = summary.stage_records.get(&1).unwrap();
        let prefill = stages.get(&Phase::Prefill).unwrap();
        let decode = stages.get(&Phase::Decode).unwrap();
        assert_eq!(prefill.records, 1);
        assert_eq!(prefill.tokens, 128);
        assert_eq!(decode.records, 1);
        assert_eq!(decode.tokens, 1);
        assert_eq!(decode.total_us, 200);
    }
}
