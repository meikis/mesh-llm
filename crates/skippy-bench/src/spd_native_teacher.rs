use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, bail};
use serde_json::json;
use skippy_runtime::spd::SpdHeadManifest;

pub struct NativeTeacherLogitsConfig<'a> {
    pub dir: PathBuf,
    pub manifest: &'a SpdHeadManifest,
    pub top_k: usize,
}

pub struct NativeTeacherLogitsWriter {
    dir: PathBuf,
    logits_f32: BufWriter<File>,
    rows_jsonl: BufWriter<File>,
    draft_token_ids: Vec<i32>,
    sample_count: usize,
    top_k: usize,
}

pub struct NativeTeacherSample<'a> {
    pub prompt_index: usize,
    pub step_index: usize,
    pub target_position: usize,
    pub query_row_index: usize,
    pub query_position: i64,
    pub target_token: i32,
    pub logits: &'a [f32],
}

impl NativeTeacherLogitsWriter {
    pub fn create(config: NativeTeacherLogitsConfig<'_>) -> Result<Self> {
        if config.top_k == 0 {
            bail!("native teacher top_k must be greater than zero");
        }
        let draft_token_ids = config
            .manifest
            .topology
            .draft_token_ids
            .as_ref()
            .context("native teacher logits require SPD manifest draft_token_ids")?
            .iter()
            .map(|token| i32::try_from(*token).context("draft token id exceeds i32"))
            .collect::<Result<Vec<_>>>()?;
        if draft_token_ids.is_empty() {
            bail!("native teacher logits require non-empty draft_token_ids");
        }
        fs::create_dir_all(&config.dir)
            .with_context(|| format!("create native teacher dir {}", config.dir.display()))?;
        let manifest = json!({
            "schema": "skippy-spd-native-teacher-logits/v1",
            "producer": "skippy-bench spd-live-tap-parity",
            "teacher_source": "native_skippy_product_verifier_current_logits",
            "native_product_teacher_logits": true,
            "paper_kl_training_ready": true,
            "logit_scope": "draft",
            "logit_width": draft_token_ids.len(),
            "top_k": config.top_k,
            "logits_tensor": {
                "path": "native_teacher_logits.f32",
                "dtype": "f32_le",
                "shape": ["sample_count", draft_token_ids.len()],
            },
            "metadata_rows": {
                "path": "native_teacher_rows.jsonl",
                "schema": "skippy-spd-native-teacher-row/v1",
            },
        });
        fs::write(
            config.dir.join("native_teacher_manifest.json"),
            format!("{}\n", serde_json::to_string_pretty(&manifest)?),
        )
        .with_context(|| {
            format!(
                "write {}",
                config.dir.join("native_teacher_manifest.json").display()
            )
        })?;
        let logits_f32 = BufWriter::new(
            File::create(config.dir.join("native_teacher_logits.f32")).with_context(|| {
                format!(
                    "create {}",
                    config.dir.join("native_teacher_logits.f32").display()
                )
            })?,
        );
        let rows_jsonl = BufWriter::new(
            File::create(config.dir.join("native_teacher_rows.jsonl")).with_context(|| {
                format!(
                    "create {}",
                    config.dir.join("native_teacher_rows.jsonl").display()
                )
            })?,
        );
        Ok(Self {
            dir: config.dir,
            logits_f32,
            rows_jsonl,
            draft_token_ids,
            sample_count: 0,
            top_k: config.top_k,
        })
    }

    pub fn draft_token_ids(&self) -> &[i32] {
        &self.draft_token_ids
    }

    pub fn write_step(&mut self, sample: NativeTeacherSample<'_>) -> Result<()> {
        if sample.logits.len() != self.draft_token_ids.len() {
            bail!(
                "native teacher logit row has width {}, expected {}",
                sample.logits.len(),
                self.draft_token_ids.len()
            );
        }
        write_f32_slice(&mut self.logits_f32, sample.logits)?;
        let target_index = self
            .draft_token_ids
            .iter()
            .position(|token| *token == sample.target_token);
        let top_k = topk(sample.logits, self.top_k);
        let topk_indices = top_k
            .iter()
            .map(|(index, _)| *index as i64)
            .collect::<Vec<_>>();
        let topk_token_ids = top_k
            .iter()
            .map(|(index, _)| i64::from(self.draft_token_ids[*index]))
            .collect::<Vec<_>>();
        let topk_logits = top_k.iter().map(|(_, logit)| *logit).collect::<Vec<_>>();
        let argmax_index = top_k.first().map(|(index, _)| *index);
        let row = json!({
            "schema": "skippy-spd-native-teacher-row/v1",
            "sample_index": self.sample_count,
            "prompt_index": sample.prompt_index,
            "step_index": sample.step_index,
            "logit_f32_offset": self.sample_count * self.draft_token_ids.len(),
            "logit_f32_count": self.draft_token_ids.len(),
            "target_position": sample.target_position,
            "query_row_index": sample.query_row_index,
            "query_position": sample.query_position,
            "target_token": sample.target_token,
            "target_logit_index": target_index.map(|index| index as i64).unwrap_or(-1),
            "label_in_logit_scope": target_index.is_some(),
            "teacher_argmax_index": argmax_index.map(|index| index as i64).unwrap_or(-1),
            "teacher_argmax_token_id": argmax_index
                .map(|index| i64::from(self.draft_token_ids[index]))
                .unwrap_or(-1),
            "teacher_top_k": {
                "indices": topk_indices,
                "token_ids": topk_token_ids,
                "logits": topk_logits,
            },
        });
        serde_json::to_writer(&mut self.rows_jsonl, &row)
            .context("write native teacher JSONL row")?;
        self.rows_jsonl
            .write_all(b"\n")
            .context("terminate native teacher JSONL row")?;
        self.sample_count += 1;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<serde_json::Value> {
        self.logits_f32
            .flush()
            .context("flush native teacher logits")?;
        self.rows_jsonl
            .flush()
            .context("flush native teacher rows")?;
        let bytes = fs::metadata(self.dir.join("native_teacher_logits.f32"))
            .with_context(|| {
                format!(
                    "stat {}",
                    self.dir.join("native_teacher_logits.f32").display()
                )
            })?
            .len();
        let summary = json!({
            "schema": "skippy-spd-native-teacher-logits-summary/v1",
            "dir": self.dir.display().to_string(),
            "sample_count": self.sample_count,
            "logit_scope": "draft",
            "logit_width": self.draft_token_ids.len(),
            "top_k": self.top_k,
            "bytes": bytes,
            "expected_bytes": self.sample_count * self.draft_token_ids.len() * std::mem::size_of::<f32>(),
            "logits_path": "native_teacher_logits.f32",
            "metadata_path": "native_teacher_rows.jsonl",
            "manifest_path": "native_teacher_manifest.json",
            "teacher_source": "native_skippy_product_verifier_current_logits",
            "native_product_teacher_logits": true,
            "paper_kl_training_ready": true,
        });
        fs::write(
            self.dir.join("native_teacher_summary.json"),
            format!("{}\n", serde_json::to_string_pretty(&summary)?),
        )
        .with_context(|| {
            format!(
                "write {}",
                self.dir.join("native_teacher_summary.json").display()
            )
        })?;
        Ok(summary)
    }
}

fn write_f32_slice(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    for value in values {
        writer
            .write_all(&value.to_le_bytes())
            .context("write f32 value")?;
    }
    Ok(())
}

fn topk(values: &[f32], limit: usize) -> Vec<(usize, f32)> {
    let mut best = Vec::with_capacity(limit.min(values.len()));
    for (index, value) in values.iter().copied().enumerate() {
        insert_topk(&mut best, (index, value), limit);
    }
    best
}

fn insert_topk(best: &mut Vec<(usize, f32)>, candidate: (usize, f32), limit: usize) {
    let insertion = best
        .iter()
        .position(|existing| topk_precedes(&candidate, existing))
        .unwrap_or(best.len());
    if insertion < limit {
        best.insert(insertion, candidate);
        best.truncate(limit);
    }
}

fn topk_precedes(left: &(usize, f32), right: &(usize, f32)) -> bool {
    left.1
        .partial_cmp(&right.1)
        .unwrap_or(std::cmp::Ordering::Less)
        .is_gt()
        || (left.1 == right.1 && left.0 < right.0)
}
