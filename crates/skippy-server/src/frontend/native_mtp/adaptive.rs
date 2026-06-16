use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::NativeMtpVerification;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::frontend) struct NativeMtpAdaptiveDisableConfig {
    pub(in crate::frontend) enabled: bool,
    pub(in crate::frontend) min_verifications: u64,
    pub(in crate::frontend) threshold: f64,
}

#[derive(Debug)]
pub(in crate::frontend) struct NativeMtpAdaptiveDisable {
    config: NativeMtpAdaptiveDisableConfig,
    accepted: u64,
    rejected: u64,
    disabled_at_verification: Option<u64>,
}

impl NativeMtpAdaptiveDisable {
    pub(in crate::frontend) fn new(config: NativeMtpAdaptiveDisableConfig) -> Self {
        Self {
            config,
            accepted: 0,
            rejected: 0,
            disabled_at_verification: None,
        }
    }

    pub(in crate::frontend) fn observe(&mut self, verification: NativeMtpVerification) -> bool {
        if !self.config.enabled || self.disabled() {
            return false;
        }
        match verification {
            NativeMtpVerification::Accepted { .. } => self.accepted += 1,
            NativeMtpVerification::Rejected { .. } => self.rejected += 1,
            NativeMtpVerification::NoPending => return false,
        }
        let verified = self.verified();
        if verified >= self.config.min_verifications && self.accept_rate() < self.config.threshold {
            self.disabled_at_verification = Some(verified);
            return true;
        }
        false
    }

    pub(in crate::frontend) fn disabled(&self) -> bool {
        self.disabled_at_verification.is_some()
    }

    pub(in crate::frontend) fn insert_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        attrs.insert(
            "llama_stage.native_mtp.adaptive_disable.enabled".to_string(),
            json!(self.config.enabled),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_disable.disabled".to_string(),
            json!(self.disabled()),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_disable.min_verifications".to_string(),
            json!(self.config.min_verifications),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_disable.threshold".to_string(),
            json!(self.config.threshold),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_disable.accept_rate".to_string(),
            json!(self.accept_rate()),
        );
        if let Some(disabled_at) = self.disabled_at_verification {
            attrs.insert(
                "llama_stage.native_mtp.adaptive_disable.disabled_at_verification".to_string(),
                json!(disabled_at),
            );
        }
    }

    fn verified(&self) -> u64 {
        self.accepted + self.rejected
    }

    fn accept_rate(&self) -> f64 {
        let verified = self.verified();
        if verified == 0 {
            0.0
        } else {
            self.accepted as f64 / verified as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_disable_triggers_after_minimum_low_acceptance_window() {
        let mut adaptive = NativeMtpAdaptiveDisable::new(NativeMtpAdaptiveDisableConfig {
            enabled: true,
            min_verifications: 4,
            threshold: 0.75,
        });

        assert!(!adaptive.observe(NativeMtpVerification::Accepted {
            draft: 1,
            target: 1,
        }));
        assert!(!adaptive.observe(NativeMtpVerification::Rejected {
            draft: 2,
            target: 3,
        }));
        assert!(!adaptive.observe(NativeMtpVerification::Accepted {
            draft: 4,
            target: 4,
        }));
        assert!(adaptive.observe(NativeMtpVerification::Rejected {
            draft: 5,
            target: 6,
        }));
        assert!(adaptive.disabled());
    }

    #[test]
    fn adaptive_disable_stays_enabled_for_high_acceptance_window() {
        let mut adaptive = NativeMtpAdaptiveDisable::new(NativeMtpAdaptiveDisableConfig {
            enabled: true,
            min_verifications: 4,
            threshold: 0.75,
        });

        for token in 0..4 {
            assert!(!adaptive.observe(NativeMtpVerification::Accepted {
                draft: token,
                target: token,
            }));
        }

        assert!(!adaptive.disabled());
    }

    #[test]
    fn adaptive_disable_ignores_no_pending_verifications() {
        let mut adaptive = NativeMtpAdaptiveDisable::new(NativeMtpAdaptiveDisableConfig {
            enabled: true,
            min_verifications: 1,
            threshold: 1.0,
        });

        assert!(!adaptive.observe(NativeMtpVerification::NoPending));
        assert!(!adaptive.disabled());
    }
}
