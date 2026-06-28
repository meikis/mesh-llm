use std::collections::BTreeMap;

use serde::Serialize;

use crate::glm_dsa_op_report::MetalDispatchRecord;

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct TimingDistributionSummary {
    pub(crate) samples: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) mean_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) min_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) p50_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) p90_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) p95_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) p99_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stdev_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) coefficient_of_variation: Option<f64>,
    pub(crate) slow_outlier_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) slow_outlier_threshold_ms: Option<f64>,
}

impl TimingDistributionSummary {
    pub(crate) fn is_empty(summary: &Self) -> bool {
        summary.samples == 0
    }
}

pub(crate) fn summarize_elapsed_ms(
    values: impl IntoIterator<Item = f64>,
) -> TimingDistributionSummary {
    let mut values: Vec<f64> = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect();
    values.sort_by(f64::total_cmp);
    let samples = values.len();
    if samples == 0 {
        return TimingDistributionSummary::default();
    }

    let sum: f64 = values.iter().sum();
    let mean = sum / samples as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / samples as f64;
    let stdev = variance.sqrt();
    let p50 = percentile(&values, 0.50);
    let slow_outlier_threshold = p50 * 1.25;
    let slow_outlier_count = values
        .iter()
        .filter(|value| **value > slow_outlier_threshold)
        .count();

    TimingDistributionSummary {
        samples,
        mean_ms: Some(mean),
        min_ms: values.first().copied(),
        p50_ms: Some(p50),
        p90_ms: Some(percentile(&values, 0.90)),
        p95_ms: Some(percentile(&values, 0.95)),
        p99_ms: Some(percentile(&values, 0.99)),
        max_ms: values.last().copied(),
        stdev_ms: Some(stdev),
        coefficient_of_variation: if mean > f64::EPSILON {
            Some(stdev / mean)
        } else {
            None
        },
        slow_outlier_count,
        slow_outlier_threshold_ms: Some(slow_outlier_threshold),
    }
}

fn percentile(sorted_values: &[f64], quantile: f64) -> f64 {
    debug_assert!(!sorted_values.is_empty());
    let last_index = sorted_values.len() - 1;
    let index = ((last_index as f64) * quantile).round() as usize;
    sorted_values[index.min(last_index)]
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct GlmDsaDispatchSummary {
    pub(crate) records: usize,
    pub(crate) topk_moe_route_fused_records: usize,
    pub(crate) topk_moe_route_pack_records: usize,
    pub(crate) topk_moe_route_encode_records: usize,
    pub(crate) dsa_sparse_attn_records: usize,
    pub(crate) mul_mat_id_records: usize,
    pub(crate) moe_weighted_sum_records: usize,
    pub(crate) moe_weighted_sum_f32x4_records: usize,
    pub(crate) routed_moe_gate_records: usize,
    pub(crate) routed_moe_up_records: usize,
    pub(crate) routed_moe_down_records: usize,
    pub(crate) routed_moe_down_q3_k_records: usize,
    pub(crate) routed_moe_down_expanded_grid_records: usize,
    pub(crate) max_grid_x: u64,
    pub(crate) max_grid_y: u64,
    pub(crate) max_grid_z: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) dispatch_shapes: Vec<DispatchShapeSummary>,
}

impl GlmDsaDispatchSummary {
    pub(crate) fn is_empty(summary: &Self) -> bool {
        summary.records == 0
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct DispatchShapeSummary {
    pub(crate) op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) kernel: Option<String>,
    pub(crate) tensor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) src_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) dst_type: Option<String>,
    pub(crate) grid_x: u64,
    pub(crate) grid_y: u64,
    pub(crate) grid_z: u64,
    pub(crate) threads_x: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) threads_y: Option<u64>,
    pub(crate) records: usize,
}

pub(crate) fn summarize_metal_dispatch(records: &[MetalDispatchRecord]) -> GlmDsaDispatchSummary {
    let mut summary = GlmDsaDispatchSummary {
        records: records.len(),
        ..GlmDsaDispatchSummary::default()
    };
    let mut shapes = BTreeMap::<DispatchShapeKey, usize>::new();

    for record in records {
        summary.max_grid_x = summary.max_grid_x.max(record.grid_x);
        summary.max_grid_y = summary.max_grid_y.max(record.grid_y);
        summary.max_grid_z = summary.max_grid_z.max(record.grid_z);

        match record.op.as_str() {
            "topk_moe_route_fused" => summary.topk_moe_route_fused_records += 1,
            "topk_moe_route_pack" => summary.topk_moe_route_pack_records += 1,
            "topk_moe_route_encode" => summary.topk_moe_route_encode_records += 1,
            "dsa_sparse_attn" => summary.dsa_sparse_attn_records += 1,
            "mul_mat_id" => summary.mul_mat_id_records += 1,
            "moe_weighted_sum" => {
                summary.moe_weighted_sum_records += 1;
                if record.kernel.as_deref() == Some("f32x4") {
                    summary.moe_weighted_sum_f32x4_records += 1;
                }
            }
            _ => {}
        }

        if record.tensor.contains("ffn_moe_gate") {
            summary.routed_moe_gate_records += 1;
        }
        if record.tensor.contains("ffn_moe_up") {
            summary.routed_moe_up_records += 1;
        }
        if record.tensor.contains("ffn_moe_down") {
            summary.routed_moe_down_records += 1;
            if record.grid_x > 256 {
                summary.routed_moe_down_expanded_grid_records += 1;
            }
            if record.src_type.as_deref() == Some("q3_K") {
                summary.routed_moe_down_q3_k_records += 1;
            }
        }

        *shapes.entry(DispatchShapeKey::from(record)).or_insert(0) += 1;
    }

    summary.dispatch_shapes = shapes
        .into_iter()
        .map(|(shape, records)| shape.into_summary(records))
        .collect();
    summary
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DispatchShapeKey {
    op: String,
    kernel: Option<String>,
    tensor: String,
    src_type: Option<String>,
    dst_type: Option<String>,
    grid_x: u64,
    grid_y: u64,
    grid_z: u64,
    threads_x: u64,
    threads_y: Option<u64>,
}

impl DispatchShapeKey {
    fn from(record: &MetalDispatchRecord) -> Self {
        Self {
            op: record.op.clone(),
            kernel: record.kernel.clone(),
            tensor: record.tensor.clone(),
            src_type: record.src_type.clone(),
            dst_type: record.dst_type.clone(),
            grid_x: record.grid_x,
            grid_y: record.grid_y,
            grid_z: record.grid_z,
            threads_x: record.threads_x,
            threads_y: record.threads_y,
        }
    }

    fn into_summary(self, records: usize) -> DispatchShapeSummary {
        DispatchShapeSummary {
            op: self.op,
            kernel: self.kernel,
            tensor: self.tensor,
            src_type: self.src_type,
            dst_type: self.dst_type,
            grid_x: self.grid_x,
            grid_y: self.grid_y,
            grid_z: self.grid_z,
            threads_x: self.threads_x,
            threads_y: self.threads_y,
            records,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_summary_reports_tail_and_outliers() {
        let summary = summarize_elapsed_ms([10.0, 11.0, 9.0, 40.0]);

        assert_eq!(summary.samples, 4);
        assert_eq!(summary.min_ms, Some(9.0));
        assert_eq!(summary.max_ms, Some(40.0));
        assert_eq!(summary.slow_outlier_count, 1);
        assert!(summary.coefficient_of_variation.unwrap() > 0.5);
    }

    #[test]
    fn metal_dispatch_summary_counts_glm_dsa_shapes() {
        let records = vec![
            dispatch("topk_moe_route_fused", None, "route", None),
            dispatch("moe_weighted_sum", Some("f32x4"), "weighted", None),
            dispatch(
                "mul_mat_id",
                None,
                "blk.45.ffn_moe_down.weight",
                Some("q3_K"),
            ),
            dispatch(
                "mul_mat_id",
                None,
                "blk.45.ffn_moe_down.weight",
                Some("q3_K"),
            ),
        ];

        let summary = summarize_metal_dispatch(&records);

        assert_eq!(summary.records, 4);
        assert_eq!(summary.topk_moe_route_fused_records, 1);
        assert_eq!(summary.mul_mat_id_records, 2);
        assert_eq!(summary.moe_weighted_sum_f32x4_records, 1);
        assert_eq!(summary.routed_moe_down_q3_k_records, 2);
        assert_eq!(summary.routed_moe_down_expanded_grid_records, 2);
        assert_eq!(summary.dispatch_shapes.len(), 3);
    }

    fn dispatch(
        op: &str,
        kernel: Option<&str>,
        tensor: &str,
        src_type: Option<&str>,
    ) -> MetalDispatchRecord {
        MetalDispatchRecord {
            op: op.to_string(),
            kernel: kernel.map(str::to_string),
            tensor: tensor.to_string(),
            parallel: None,
            q_type: None,
            k_type: None,
            v_type: None,
            mask_type: None,
            top_k_type: None,
            src_type: src_type.map(str::to_string),
            dst_type: None,
            q_width: None,
            v_width: None,
            batch: None,
            heads: None,
            stream: None,
            kv: None,
            top_k: None,
            top_stream: None,
            grid_x: if tensor.contains("ffn_moe_down") {
                1536
            } else {
                256
            },
            grid_y: 1,
            grid_z: 8,
            threads_x: 32,
            threads_y: Some(2),
        }
    }
}
