use crate::gguf::GgufCompactMeta;

const GLM_DSA_ARCHITECTURE: &str = "glm-dsa";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GgufKvCacheType {
    F16,
    Q8_0,
    Q4_0,
}

impl GgufKvCacheType {
    pub fn from_llama_arg(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "f16" => Some(Self::F16),
            "q8_0" => Some(Self::Q8_0),
            "q4_0" => Some(Self::Q4_0),
            _ => None,
        }
    }

    pub const fn as_llama_arg(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    const fn block_shape(self) -> (u64, u64) {
        match self {
            Self::F16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Q4_0 => (32, 18),
        }
    }

    fn bytes_for_elements(self, elements: u64) -> Option<u64> {
        let (block_elements, block_bytes) = self.block_shape();
        let blocks = elements
            .checked_add(block_elements.checked_sub(1)?)?
            .checked_div(block_elements)?;
        blocks.checked_mul(block_bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GgufKvCacheQuant {
    pub k: GgufKvCacheType,
    pub v: GgufKvCacheType,
}

impl GgufKvCacheQuant {
    /// f16 K + f16 V: highest quality, largest KV cache.
    pub const F16: Self = Self {
        k: GgufKvCacheType::F16,
        v: GgufKvCacheType::F16,
    };

    /// q8_0 K + q8_0 V: moderate compression.
    pub const Q8_0: Self = Self {
        k: GgufKvCacheType::Q8_0,
        v: GgufKvCacheType::Q8_0,
    };

    /// q4_0 K + q4_0 V: most aggressive compression, smallest KV cache.
    pub const Q4_0: Self = Self {
        k: GgufKvCacheType::Q4_0,
        v: GgufKvCacheType::Q4_0,
    };

    pub const fn new(k: GgufKvCacheType, v: GgufKvCacheType) -> Self {
        Self { k, v }
    }

    pub const fn f16() -> Self {
        Self::F16
    }

    /// Returns `true` if `self` uses more aggressive (smaller) quantisation
    /// than `other`.
    pub const fn is_more_aggressive_than(self, other: Self) -> bool {
        Self::aggressiveness(self) > Self::aggressiveness(other)
    }

    const fn aggressiveness(q: Self) -> u8 {
        Self::type_aggressiveness(q.k) + Self::type_aggressiveness(q.v)
    }

    const fn type_aggressiveness(t: GgufKvCacheType) -> u8 {
        match t {
            GgufKvCacheType::F16 => 0,
            GgufKvCacheType::Q8_0 => 1,
            GgufKvCacheType::Q4_0 => 2,
        }
    }

    pub fn from_llama_args(cache_type_k: &str, cache_type_v: &str) -> Option<Self> {
        Some(Self {
            k: GgufKvCacheType::from_llama_arg(cache_type_k)?,
            v: GgufKvCacheType::from_llama_arg(cache_type_v)?,
        })
    }

    pub fn k_cache_bytes_per_token(self, meta: &GgufCompactMeta) -> Option<u64> {
        if meta.architecture == GLM_DSA_ARCHITECTURE {
            return glm_dsa_k_cache_bytes_per_token(meta, self.k);
        }
        standard_cache_bytes_per_token(meta, meta.key_length, self.k)
    }

    pub fn v_cache_bytes_per_token(self, meta: &GgufCompactMeta) -> Option<u64> {
        if meta.architecture == GLM_DSA_ARCHITECTURE {
            return glm_dsa_compressed_key_length(meta).map(|_| 0);
        }
        standard_cache_bytes_per_token(meta, meta.value_length, self.v)
    }

    pub fn kv_cache_bytes_per_token(self, meta: &GgufCompactMeta) -> Option<u64> {
        self.k_cache_bytes_per_token(meta)?
            .checked_add(self.v_cache_bytes_per_token(meta)?)
    }
}

fn glm_dsa_k_cache_bytes_per_token(
    meta: &GgufCompactMeta,
    cache_type: GgufKvCacheType,
) -> Option<u64> {
    let main_cache = cache_vector_bytes_per_token(
        meta.layer_count,
        glm_dsa_compressed_key_length(meta)?,
        cache_type,
    )?;
    let indexer_cache = cache_vector_bytes_per_token(
        meta.layer_count,
        nonzero(meta.indexer_key_length)?,
        cache_type,
    )?;
    main_cache.checked_add(indexer_cache)
}

fn glm_dsa_compressed_key_length(meta: &GgufCompactMeta) -> Option<u32> {
    nonzero(meta.kv_lora_rank)?.checked_add(nonzero(meta.rope_dimension_count)?)
}

fn standard_cache_bytes_per_token(
    meta: &GgufCompactMeta,
    vector_length: u32,
    cache_type: GgufKvCacheType,
) -> Option<u64> {
    let kv_heads = u64::from(meta.effective_kv_head_count()?);
    let elements_per_layer = kv_heads.checked_mul(u64::from(nonzero(vector_length)?))?;
    cache_vector_bytes_per_token(meta.layer_count, elements_per_layer, cache_type)
}

fn cache_vector_bytes_per_token(
    layer_count: u32,
    elements_per_layer: impl Into<u64>,
    cache_type: GgufKvCacheType,
) -> Option<u64> {
    cache_type
        .bytes_for_elements(elements_per_layer.into())?
        .checked_mul(u64::from(nonzero(layer_count)?))
}

fn nonzero(value: u32) -> Option<u32> {
    (value > 0).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_cache_prices_key_and_value_types_independently() {
        let meta = GgufCompactMeta {
            head_count: 32,
            kv_head_count: 8,
            layer_count: 24,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        };
        let quant = GgufKvCacheQuant::new(GgufKvCacheType::Q8_0, GgufKvCacheType::Q4_0);

        assert_eq!(quant.k_cache_bytes_per_token(&meta), Some(26_112));
        assert_eq!(quant.v_cache_bytes_per_token(&meta), Some(13_824));
        assert_eq!(quant.kv_cache_bytes_per_token(&meta), Some(39_936));
    }

    #[test]
    fn standard_cache_prices_key_and_value_widths_independently() {
        let meta = GgufCompactMeta {
            head_count: 32,
            kv_head_count: 8,
            layer_count: 24,
            key_length: 64,
            value_length: 256,
            ..Default::default()
        };
        let quant = GgufKvCacheQuant::new(GgufKvCacheType::Q8_0, GgufKvCacheType::Q4_0);

        assert_eq!(quant.k_cache_bytes_per_token(&meta), Some(13_056));
        assert_eq!(quant.v_cache_bytes_per_token(&meta), Some(27_648));
        assert_eq!(quant.kv_cache_bytes_per_token(&meta), Some(40_704));
    }

    #[test]
    fn glm_dsa_prices_compressed_mla_and_indexer_keys_without_value_cache() {
        let meta = GgufCompactMeta {
            architecture: GLM_DSA_ARCHITECTURE.to_string(),
            head_count: 64,
            kv_head_count: 64,
            layer_count: 79,
            key_length: 576,
            value_length: 256,
            kv_lora_rank: 512,
            rope_dimension_count: 64,
            indexer_key_length: 128,
            ..Default::default()
        };

        assert_eq!(
            GgufKvCacheQuant::Q4_0.k_cache_bytes_per_token(&meta),
            Some(31_284)
        );
        assert_eq!(
            GgufKvCacheQuant::Q4_0.v_cache_bytes_per_token(&meta),
            Some(0)
        );
        assert_eq!(
            GgufKvCacheQuant::Q4_0.kv_cache_bytes_per_token(&meta),
            Some(31_284)
        );
    }

    #[test]
    fn required_cache_metadata_must_be_present() {
        let meta = GgufCompactMeta {
            head_count: 32,
            layer_count: 24,
            key_length: 128,
            ..Default::default()
        };

        assert_eq!(meta.k_cache_bytes_per_token_f16(), Some(196_608));
        assert_eq!(meta.v_cache_bytes_per_token_f16(), None);
        assert_eq!(
            GgufKvCacheQuant::f16().kv_cache_bytes_per_token(&meta),
            None
        );
    }
}
