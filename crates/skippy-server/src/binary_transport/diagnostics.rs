const INDEXSHARE_EXEC_LOG_ENV: &str = "LLAMA_GLM_DSA_INDEXSHARE_EXEC_LOG";

pub(crate) fn glm_dsa_indexshare_exec_log_enabled() -> bool {
    std::env::var(INDEXSHARE_EXEC_LOG_ENV)
        .ok()
        .is_some_and(|value| diagnostic_flag_enabled(&value))
}

fn diagnostic_flag_enabled(value: &str) -> bool {
    value.parse::<i64>().is_ok_and(|value| value != 0)
}

#[cfg(test)]
mod tests {
    use super::diagnostic_flag_enabled;

    #[test]
    fn indexshare_log_flag_matches_numeric_llama_environment_semantics() {
        assert!(!diagnostic_flag_enabled(""));
        assert!(!diagnostic_flag_enabled("0"));
        assert!(!diagnostic_flag_enabled("true"));
        assert!(diagnostic_flag_enabled("1"));
        assert!(diagnostic_flag_enabled("-1"));
    }
}
