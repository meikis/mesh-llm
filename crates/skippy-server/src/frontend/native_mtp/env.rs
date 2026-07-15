const REJECT_COOLDOWN_TOKENS_ENV: &str = "SKIPPY_NATIVE_MTP_REJECT_COOLDOWN_TOKENS";
const SUPPRESS_COOLDOWN_DRAFTS_ENV: &str = "SKIPPY_NATIVE_MTP_SUPPRESS_COOLDOWN_DRAFTS";
const SUPPRESS_COOLDOWN_DRAFT_LIMIT_ENV: &str = "SKIPPY_NATIVE_MTP_SUPPRESS_COOLDOWN_DRAFT_LIMIT";
const NGRAM_HYBRID_ENV: &str = "SKIPPY_NATIVE_MTP_NGRAM_HYBRID";
const NGRAM_SIZE_ENV: &str = "SKIPPY_NATIVE_MTP_NGRAM_SIZE";
const NGRAM_MAX_PROPOSAL_TOKENS_ENV: &str = "SKIPPY_NATIVE_MTP_NGRAM_MAX_PROPOSAL_TOKENS";

pub(in crate::frontend) fn native_mtp_reject_cooldown_tokens() -> usize {
    parse_usize_env(REJECT_COOLDOWN_TOKENS_ENV, 0)
}

pub(in crate::frontend) fn native_mtp_suppress_cooldown_drafts_enabled() -> bool {
    truthy_env(std::env::var(SUPPRESS_COOLDOWN_DRAFTS_ENV).ok().as_deref())
}

pub(in crate::frontend) fn native_mtp_suppress_cooldown_draft_limit() -> usize {
    parse_usize_env(SUPPRESS_COOLDOWN_DRAFT_LIMIT_ENV, 0)
}

pub(in crate::frontend) fn native_mtp_ngram_hybrid_enabled() -> bool {
    truthy_env(std::env::var(NGRAM_HYBRID_ENV).ok().as_deref())
}

pub(in crate::frontend) fn native_mtp_ngram_size() -> usize {
    parse_usize_env(NGRAM_SIZE_ENV, 8)
}

pub(in crate::frontend) fn native_mtp_ngram_max_proposal_tokens() -> usize {
    parse_usize_env(NGRAM_MAX_PROPOSAL_TOKENS_ENV, 4)
}

fn truthy_env(value: Option<&str>) -> bool {
    matches!(
        normalized_env(value).as_deref(),
        Some("1" | "true" | "on" | "enable" | "enabled" | "yes")
    )
}

fn normalized_env(value: Option<&str>) -> Option<String> {
    value.map(str::trim).map(str::to_ascii_lowercase)
}

fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truthy_env_accepts_enabled_aliases_only() {
        for value in ["1", "true", " enabled ", "yes", "on"] {
            assert!(truthy_env(Some(value)), "{value}");
        }
        for value in [
            None,
            Some("0"),
            Some("false"),
            Some("off"),
            Some("disabled"),
        ] {
            assert!(!truthy_env(value), "{value:?}");
        }
    }

    #[test]
    fn numeric_options_default_when_absent() {
        assert_eq!(parse_usize_env("SKIPPY_TEST_MISSING_REJECT_COOLDOWN", 0), 0);
        assert_eq!(
            parse_usize_env("SKIPPY_TEST_MISSING_SUPPRESS_COOLDOWN_LIMIT", 0),
            0
        );
    }
}
