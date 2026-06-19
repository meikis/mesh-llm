use std::{fs, path::Path};

use super::remote::CaseDeployment;

pub(super) fn stage_log_tail_summary(deployment: &CaseDeployment, line_count: usize) -> String {
    let mut summary = format!("stage log tails after readiness failure (last {line_count} lines)");
    for (index, stage) in deployment.stages.iter().enumerate() {
        summary.push_str(&format!(
            "\n\n--- stage{index}: {} ---\n",
            stage.log_path.display()
        ));
        summary.push_str(&tail_file(&stage.log_path, line_count));
    }
    summary
}

fn tail_file(path: &Path, line_count: usize) -> String {
    match fs::read_to_string(path) {
        Ok(content) => tail_lines(&content, line_count),
        Err(error) => format!("unable to read stage log: {error}"),
    }
}

fn tail_lines(content: &str, line_count: usize) -> String {
    let lines = content.lines().collect::<Vec<_>>();
    let start = lines.len().saturating_sub(line_count);
    let tail = lines[start..].join("\n");
    if tail.is_empty() {
        "stage log is empty".to_string()
    } else {
        tail
    }
}
