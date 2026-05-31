use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{cmp::Ordering, env, fs, path::PathBuf};

const DEFAULT_SCENARIO: &str = "steady_decode";
const DEFAULT_MAX_MEDIAN_ABSOLUTE_ERROR: f64 = 0.10;
const DEFAULT_MAX_INDIVIDUAL_ERROR: f64 = 0.15;
const DEFAULT_MAX_NOISY: usize = 0;

#[derive(Debug)]
struct Args {
    report_json: PathBuf,
    scenario: String,
    max_median_absolute_error: f64,
    max_individual_error: f64,
    max_noisy: usize,
    min_models: Option<usize>,
    markdown_out: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct ValidationReport {
    models: Vec<ModelReport>,
    summary: Option<ValidationSummary>,
}

#[derive(Debug, Deserialize)]
struct ValidationSummary {
    error_count: usize,
}

#[derive(Debug, Deserialize)]
struct ModelReport {
    input_ref: String,
    benchmarks: Vec<ScenarioReport>,
    errors: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ScenarioReport {
    scenario: String,
    predicted: Option<f64>,
    observed: Option<f64>,
    observed_over_fit: Option<f64>,
    verdict: String,
    benchmark: BenchmarkSummary,
}

#[derive(Debug, Deserialize)]
struct BenchmarkSummary {
    spread_pct: Option<f64>,
    sample_count: usize,
    errors: Vec<String>,
}

#[derive(Debug)]
struct ScenarioRow<'a> {
    model: &'a ModelReport,
    scenario: &'a ScenarioReport,
    absolute_error: f64,
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let bytes = fs::read(&args.report_json)
        .with_context(|| format!("read validation report {}", args.report_json.display()))?;
    let report: ValidationReport =
        serde_json::from_slice(&bytes).context("parse validation report JSON")?;
    let rows = scenario_rows(&report, &args.scenario);
    let markdown = render_markdown(&args, &report, &rows);
    print!("{markdown}");
    if let Some(path) = &args.markdown_out {
        fs::write(path, &markdown)
            .with_context(|| format!("write markdown report {}", path.display()))?;
    }
    enforce_thresholds(&args, &report, &rows)
}

impl Args {
    fn parse() -> Result<Self> {
        let mut values = env::args().skip(1);
        let mut parsed = Self {
            report_json: PathBuf::new(),
            scenario: DEFAULT_SCENARIO.into(),
            max_median_absolute_error: DEFAULT_MAX_MEDIAN_ABSOLUTE_ERROR,
            max_individual_error: DEFAULT_MAX_INDIVIDUAL_ERROR,
            max_noisy: DEFAULT_MAX_NOISY,
            min_models: None,
            markdown_out: None,
        };

        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--scenario" => parsed.scenario = next_value(&mut values, "--scenario")?,
                "--max-median-absolute-error" => {
                    parsed.max_median_absolute_error =
                        parse_next(&mut values, "--max-median-absolute-error")?;
                }
                "--max-individual-error" => {
                    parsed.max_individual_error =
                        parse_next(&mut values, "--max-individual-error")?;
                }
                "--max-noisy" => parsed.max_noisy = parse_next(&mut values, "--max-noisy")?,
                "--min-models" => {
                    parsed.min_models = Some(parse_next(&mut values, "--min-models")?)
                }
                "--markdown-out" => {
                    parsed.markdown_out =
                        Some(PathBuf::from(next_value(&mut values, "--markdown-out")?));
                }
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                other if other.starts_with('-') => bail!("unknown argument {other}"),
                path => {
                    if !parsed.report_json.as_os_str().is_empty() {
                        bail!("only one report JSON path may be provided");
                    }
                    parsed.report_json = PathBuf::from(path);
                }
            }
        }

        if parsed.report_json.as_os_str().is_empty() {
            bail!("provide a validation report JSON path");
        }
        Ok(parsed)
    }
}

fn scenario_rows<'a>(report: &'a ValidationReport, scenario: &str) -> Vec<ScenarioRow<'a>> {
    report
        .models
        .iter()
        .filter_map(|model| {
            let scenario = model
                .benchmarks
                .iter()
                .find(|benchmark| benchmark.scenario == scenario)?;
            let ratio = scenario.observed_over_fit?;
            Some(ScenarioRow {
                model,
                scenario,
                absolute_error: (ratio - 1.0).abs(),
            })
        })
        .collect()
}

fn render_markdown(args: &Args, report: &ValidationReport, rows: &[ScenarioRow<'_>]) -> String {
    let median_error = median_absolute_error(rows);
    let noisy = noisy_count(rows);
    let mut markdown = String::new();
    markdown.push_str("# Model Fit Validation\n\n");
    markdown.push_str(&format!("Scenario: `{}`\n\n", args.scenario));
    markdown.push_str("| metric | value |\n|---|---:|\n");
    markdown.push_str(&format!("| models in report | {} |\n", report.models.len()));
    markdown.push_str(&format!("| scenario samples | {} |\n", rows.len()));
    markdown.push_str(&format!("| noisy samples | {noisy} |\n"));
    markdown.push_str(&format!(
        "| median absolute error | {} |\n\n",
        percent_option(median_error)
    ));
    markdown.push_str(
        "| model | predicted tok/s | observed tok/s | observed/fit | spread | samples | verdict |\n",
    );
    markdown.push_str("|---|---:|---:|---:|---:|---:|---|\n");
    for row in rows {
        markdown.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {} | {} |\n",
            row.model.input_ref,
            number_option(row.scenario.predicted),
            number_option(row.scenario.observed),
            ratio_option(row.scenario.observed_over_fit),
            percent_option(row.scenario.benchmark.spread_pct.map(|value| value / 100.0)),
            row.scenario.benchmark.sample_count,
            row.scenario.verdict
        ));
    }
    markdown
}

fn enforce_thresholds(
    args: &Args,
    report: &ValidationReport,
    rows: &[ScenarioRow<'_>],
) -> Result<()> {
    let mut failures = Vec::new();
    if let Some(summary) = &report.summary
        && summary.error_count > 0
    {
        failures.push(format!(
            "validation report contains {} model-level errors",
            summary.error_count
        ));
    }
    for model in &report.models {
        if !model.errors.is_empty() {
            failures.push(format!(
                "{} has model errors: {}",
                model.input_ref,
                model.errors.join("; ")
            ));
        }
    }
    if let Some(min_models) = args.min_models
        && rows.len() < min_models
    {
        failures.push(format!(
            "scenario {} produced {} samples, expected at least {min_models}",
            args.scenario,
            rows.len()
        ));
    }
    let noisy = noisy_count(rows);
    if noisy > args.max_noisy {
        failures.push(format!(
            "scenario {} had {noisy} noisy samples, max allowed is {}",
            args.scenario, args.max_noisy
        ));
    }
    if let Some(median_error) = median_absolute_error(rows)
        && median_error > args.max_median_absolute_error
    {
        failures.push(format!(
            "median absolute error {:.2}% exceeded {:.2}%",
            median_error * 100.0,
            args.max_median_absolute_error * 100.0
        ));
    }
    for row in rows {
        if row.absolute_error > args.max_individual_error {
            failures.push(format!(
                "{} scenario {} error {:.2}% exceeded {:.2}%",
                row.model.input_ref,
                args.scenario,
                row.absolute_error * 100.0,
                args.max_individual_error * 100.0
            ));
        }
        if !row.scenario.benchmark.errors.is_empty() {
            failures.push(format!(
                "{} scenario {} benchmark errors: {}",
                row.model.input_ref,
                args.scenario,
                row.scenario.benchmark.errors.join("; ")
            ));
        }
    }

    if failures.is_empty() {
        return Ok(());
    }
    bail!("model-fit validation failed:\n{}", failures.join("\n"))
}

fn noisy_count(rows: &[ScenarioRow<'_>]) -> usize {
    rows.iter()
        .filter(|row| row.scenario.verdict == "inconclusive-noisy")
        .count()
}

fn median_absolute_error(rows: &[ScenarioRow<'_>]) -> Option<f64> {
    let mut samples = rows
        .iter()
        .map(|row| row.absolute_error)
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let mid = samples.len() / 2;
    Some(if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    })
}

fn number_option(value: Option<f64>) -> String {
    value.map_or_else(|| "-".into(), |value| format!("{value:.1}"))
}

fn ratio_option(value: Option<f64>) -> String {
    value.map_or_else(|| "-".into(), |value| format!("{value:.3}"))
}

fn percent_option(value: Option<f64>) -> String {
    value.map_or_else(|| "-".into(), |value| format!("{:.1}%", value * 100.0))
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    next_value(args, name)?
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid {name}: {err}"))
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))
}

fn print_usage() {
    eprintln!(
        "usage: model-fit-check-validation [--scenario steady_decode] [--max-median-absolute-error 0.10] [--max-individual-error 0.15] [--max-noisy 0] [--min-models N] [--markdown-out report.md] report.json"
    );
}
