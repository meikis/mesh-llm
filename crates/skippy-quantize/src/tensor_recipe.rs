use anyhow::{Result, anyhow};

use crate::quantize::normalize_tensor_type_entry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuiltinTensorRecipe {
    GlmDsaQ2KMtpQ8,
    GlmDsaUdQ3KSMtpQ8,
}

impl BuiltinTensorRecipe {
    pub(crate) fn parse(label: &str) -> Result<Self> {
        match label {
            "glm-dsa-q2-k-mtp-q8" => Ok(Self::GlmDsaQ2KMtpQ8),
            "glm-dsa-ud-q3-k-s-mtp-q8" => Ok(Self::GlmDsaUdQ3KSMtpQ8),
            _ => Err(anyhow!("unsupported built-in tensor recipe {label:?}")),
        }
    }

    pub(crate) fn entries(self) -> &'static [&'static str] {
        match self {
            Self::GlmDsaQ2KMtpQ8 => GLM_DSA_Q2_K_MTP_Q8,
            Self::GlmDsaUdQ3KSMtpQ8 => GLM_DSA_UD_Q3_K_S_MTP_Q8,
        }
    }

    pub(crate) fn normalized_entries(self) -> Result<Vec<String>> {
        self.entries()
            .iter()
            .map(|entry| normalize_tensor_type_entry(entry))
            .collect()
    }
}

const GLM_DSA_Q2_K_MTP_Q8: &[&str] = &[
    "^token_embd\\.weight$=Q8_0",
    "^output\\.weight$=Q8_0",
    "(^|\\.)nextn\\.=Q8_0",
    "^d2t\\.weight$=Q8_0",
    "\\.shared_head_(head|norm)\\.weight$=Q8_0",
    "\\.attn_(q_a|q_b|kv_a_mqa|kv_b)\\.weight$=Q8_0",
    "\\.indexer\\.k_norm\\.bias$=F32",
    "\\.indexer\\.(k_norm|proj|attn_k|attn_q_b)\\.weight$=Q8_0",
    "\\.ffn_gate_inp(_shexp)?\\.weight$=Q8_0",
    "\\.ffn_(gate|up|down)_shexp\\.weight$=Q4_K",
];

const GLM_DSA_UD_Q3_K_S_MTP_Q8: &[&str] = &[
    "^token_embd\\.weight$=Q8_0",
    "^output\\.weight$=Q8_0",
    "(^|\\.)nextn\\.=Q8_0",
    "^d2t\\.weight$=Q8_0",
    "\\.shared_head_(head|norm)\\.weight$=Q8_0",
    "\\.attn_(q_a|q_b|kv_a_mqa|kv_b)\\.weight$=Q8_0",
    "\\.indexer\\.k_norm\\.bias$=F32",
    "\\.indexer\\.(k_norm|proj|attn_k|attn_q_b)\\.weight$=Q8_0",
    "\\.ffn_gate_inp(_shexp)?\\.weight$=Q8_0",
    "\\.ffn_(gate|up|down)_shexp\\.weight$=Q4_K",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_recipes() {
        assert_eq!(
            BuiltinTensorRecipe::parse("glm-dsa-q2-k-mtp-q8").unwrap(),
            BuiltinTensorRecipe::GlmDsaQ2KMtpQ8
        );
        assert!(BuiltinTensorRecipe::parse("unknown").is_err());
    }

    #[test]
    fn built_in_entries_are_valid_tensor_overrides() {
        for recipe in [
            BuiltinTensorRecipe::GlmDsaQ2KMtpQ8,
            BuiltinTensorRecipe::GlmDsaUdQ3KSMtpQ8,
        ] {
            let entries = recipe.normalized_entries().unwrap();
            assert_eq!(entries.len(), recipe.entries().len());
            assert!(entries.iter().any(|entry| entry.contains("nextn")));
            assert!(entries.iter().any(|entry| entry.contains("indexer")));
            assert!(entries.iter().any(|entry| entry.ends_with("=Q8_0")));
        }
    }
}
