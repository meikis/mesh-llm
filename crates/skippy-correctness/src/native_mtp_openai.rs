use std::{
    fs::{self, File},
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::{
    cli::{NativeMtpOpenAiAbArgs, StageLoadMode},
    report::{NativeMtpOpenAiAbReport, NativeMtpOpenAiCaseReport, NativeMtpOpenAiMetricsReport},
    support::{ChildGuard, connect_ready, generate_run_id},
};

const BATCHED_VERIFY_ENV: &str = "SKIPPY_NATIVE_MTP_BATCHED_VERIFY";
const NATIVE_MTP_ENABLED_ENV: &str = "SKIPPY_NATIVE_MTP_ENABLED";

struct OpenAiCaseConfig {
    case: &'static str,
    native_mtp_enabled: bool,
    batched_verify_enabled: bool,
    run_id: String,
    root: PathBuf,
    openai_bind_addr: SocketAddr,
    stage0_bind_addr: SocketAddr,
    stage0_endpoint_addr: SocketAddr,
    stage1_bind_addr: SocketAddr,
    stage1_endpoint_addr: SocketAddr,
}

struct OpenAiStageConfig<'a> {
    run_id: &'a str,
    model_id: &'a str,
    model_path: &'a Path,
    stage_id: &'a str,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    bind_addr: SocketAddr,
    upstream: Option<Value>,
    downstream: Option<Value>,
}

struct Stage1LaunchReport {
    launch_mode: &'static str,
    remote_config: Option<String>,
    remote_topology: Option<String>,
    remote_log: Option<String>,
}

enum StageHandle {
    Local(ChildGuard),
    Remote(RemoteStageGuard),
    External,
}

struct RemoteStageGuard {
    host: String,
    pid: Option<u32>,
    remote_log: String,
}

pub fn native_mtp_openai_ab(args: NativeMtpOpenAiAbArgs) -> Result<()> {
    if args.runtime.stage_load_mode != StageLoadMode::RuntimeSlice {
        bail!("native-mtp-open-ai-ab currently supports --stage-load-mode runtime-slice only");
    }
    if args.split_layer == 0 || args.split_layer >= args.runtime.layer_end {
        bail!("split_layer must be greater than zero and less than layer_end");
    }
    if args.external_stage1 && args.stage1_ssh_host.is_some() {
        bail!("--external-stage1 cannot be combined with --stage1-ssh-host");
    }
    if args.stage1_ssh_host.is_some() && args.stage1_remote_stage_server_bin.is_none() {
        bail!("--stage1-remote-stage-server-bin is required with --stage1-ssh-host");
    }

    let model_id = args.runtime.model_id.clone().unwrap_or_else(|| {
        args.runtime
            .model
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("local-model")
            .to_string()
    });
    let root = args
        .case_root
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join(generate_run_id()));
    fs::create_dir_all(&root).with_context(|| format!("failed to create {}", root.display()))?;
    let client = Client::builder()
        .timeout(Duration::from_secs(args.request_timeout_secs.max(1)))
        .build()
        .context("failed to build HTTP client")?;

    let baseline = run_openai_case(
        &args,
        &client,
        &model_id,
        OpenAiCaseConfig {
            case: "baseline",
            native_mtp_enabled: false,
            batched_verify_enabled: false,
            run_id: format!("{}-baseline", generate_run_id()),
            root: root.join("baseline"),
            openai_bind_addr: case_addr(args.openai_bind_addr, args.batched_port_offset, 0)?,
            stage0_bind_addr: case_addr(args.stage0_bind_addr, args.batched_port_offset, 0)?,
            stage0_endpoint_addr: case_addr(
                args.stage0_endpoint_addr.unwrap_or(args.stage0_bind_addr),
                args.batched_port_offset,
                0,
            )?,
            stage1_bind_addr: case_addr(args.stage1_bind_addr, args.batched_port_offset, 0)?,
            stage1_endpoint_addr: case_addr(
                args.stage1_endpoint_addr.unwrap_or(args.stage1_bind_addr),
                args.batched_port_offset,
                0,
            )?,
        },
    )?;
    let n1 = run_openai_case(
        &args,
        &client,
        &model_id,
        OpenAiCaseConfig {
            case: "n1",
            native_mtp_enabled: true,
            batched_verify_enabled: false,
            run_id: format!("{}-n1", generate_run_id()),
            root: root.join("n1"),
            openai_bind_addr: case_addr(args.openai_bind_addr, args.batched_port_offset, 1)?,
            stage0_bind_addr: case_addr(args.stage0_bind_addr, args.batched_port_offset, 1)?,
            stage0_endpoint_addr: case_addr(
                args.stage0_endpoint_addr.unwrap_or(args.stage0_bind_addr),
                args.batched_port_offset,
                1,
            )?,
            stage1_bind_addr: case_addr(args.stage1_bind_addr, args.batched_port_offset, 1)?,
            stage1_endpoint_addr: case_addr(
                args.stage1_endpoint_addr.unwrap_or(args.stage1_bind_addr),
                args.batched_port_offset,
                1,
            )?,
        },
    )?;
    let batched = run_openai_case(
        &args,
        &client,
        &model_id,
        OpenAiCaseConfig {
            case: "batched",
            native_mtp_enabled: true,
            batched_verify_enabled: true,
            run_id: format!("{}-batched", generate_run_id()),
            root: root.join("batched"),
            openai_bind_addr: case_addr(args.openai_bind_addr, args.batched_port_offset, 2)?,
            stage0_bind_addr: case_addr(args.stage0_bind_addr, args.batched_port_offset, 2)?,
            stage0_endpoint_addr: case_addr(
                args.stage0_endpoint_addr.unwrap_or(args.stage0_bind_addr),
                args.batched_port_offset,
                2,
            )?,
            stage1_bind_addr: case_addr(args.stage1_bind_addr, args.batched_port_offset, 2)?,
            stage1_endpoint_addr: case_addr(
                args.stage1_endpoint_addr.unwrap_or(args.stage1_bind_addr),
                args.batched_port_offset,
                2,
            )?,
        },
    )?;

    let exact_content_match = baseline.content == n1.content && baseline.content == batched.content;
    let batched_events_present = batched.metrics.batched_verify_events > 0;
    let require_batched_events = !args.allow_missing_batched_events;
    let matches = baseline.http_status == 200
        && n1.http_status == 200
        && batched.http_status == 200
        && exact_content_match
        && !baseline.metrics.native_mtp_enabled
        && n1.metrics.native_mtp_enabled
        && batched.metrics.native_mtp_enabled
        && baseline.metrics.fatal_error_events == 0
        && n1.metrics.fatal_error_events == 0
        && batched.metrics.fatal_error_events == 0
        && (!require_batched_events || batched_events_present);

    let report = NativeMtpOpenAiAbReport {
        mode: "native-mtp-open-ai-ab",
        status: status(matches),
        model_id,
        model_path: args.runtime.model.display().to_string(),
        prompt: args.runtime.prompt,
        max_tokens: args.max_tokens,
        split_layer: args.split_layer,
        layer_end: args.runtime.layer_end,
        activation_width: args.activation_width,
        activation_wire_dtype: args.activation_wire_dtype,
        exact_content_match,
        batched_events_required: require_batched_events,
        batched_events_present,
        matches,
        baseline,
        n1,
        batched,
    };
    emit_report(&report, args.output.report_out.as_deref())?;
    if !report.matches && !args.allow_mismatch {
        bail!("native MTP OpenAI n=1 and batched verification did not match");
    }
    Ok(())
}

fn run_openai_case(
    args: &NativeMtpOpenAiAbArgs,
    client: &Client,
    model_id: &str,
    case: OpenAiCaseConfig,
) -> Result<NativeMtpOpenAiCaseReport> {
    fs::create_dir_all(&case.root)
        .with_context(|| format!("failed to create {}", case.root.display()))?;
    let stage0_config_path = case.root.join("stage0.json");
    let stage1_config_path = case.root.join("stage1.json");
    let topology_path = case.root.join("topology.json");
    let stage0_log = case.root.join("stage0.log");
    let stage1_log = case.root.join("stage1.log");
    let stage0_model_path = args.stage0_model.as_deref().unwrap_or(&args.runtime.model);
    let stage1_model_path = args.stage1_model.as_deref().unwrap_or(&args.runtime.model);

    write_stage_config(
        &stage0_config_path,
        &stage_config_json(
            args,
            OpenAiStageConfig {
                run_id: &case.run_id,
                model_id,
                model_path: stage0_model_path,
                stage_id: "stage-0",
                stage_index: 0,
                layer_start: 0,
                layer_end: args.split_layer,
                bind_addr: case.stage0_bind_addr,
                upstream: None,
                downstream: Some(json!({
                    "stage_id": "stage-1",
                    "stage_index": 1,
                    "endpoint": format!("tcp://{}", case.stage1_endpoint_addr),
                })),
            },
        ),
    )?;
    write_stage_config(
        &stage1_config_path,
        &stage_config_json(
            args,
            OpenAiStageConfig {
                run_id: &case.run_id,
                model_id,
                model_path: stage1_model_path,
                stage_id: "stage-1",
                stage_index: 1,
                layer_start: args.split_layer,
                layer_end: args.runtime.layer_end,
                bind_addr: case.stage1_bind_addr,
                upstream: Some(json!({
                    "stage_id": "stage-0",
                    "stage_index": 0,
                    "endpoint": format!("tcp://{}", case.stage0_endpoint_addr),
                })),
                downstream: None,
            },
        ),
    )?;
    write_stage_config(
        &topology_path,
        &json!({
            "topology_id": "native-mtp-open-ai-ab",
            "model_id": model_id,
            "stages": [
                {
                    "stage_id": "stage-0",
                    "stage_index": 0,
                    "host": "localhost",
                    "endpoint": format!("tcp://{}", case.stage0_endpoint_addr),
                    "layer_start": 0,
                    "layer_end": args.split_layer,
                    "load_mode": "runtime-slice",
                },
                {
                    "stage_id": "stage-1",
                    "stage_index": 1,
                    "host": "localhost",
                    "endpoint": format!("tcp://{}", case.stage1_endpoint_addr),
                    "layer_start": args.split_layer,
                    "layer_end": args.runtime.layer_end,
                    "load_mode": "runtime-slice",
                },
            ],
        }),
    )?;

    let (stage1, stage1_launch) = start_stage1(
        args,
        &case.run_id,
        &stage1_config_path,
        &topology_path,
        &stage1_log,
        case.native_mtp_enabled,
        case.batched_verify_enabled,
    )?;
    drop(
        connect_ready(case.stage1_endpoint_addr, args.server.startup_timeout_secs)
            .context("stage 1 binary server did not become ready")?,
    );
    let _stage0 = spawn_stage(
        args,
        &stage0_config_path,
        Some(case.openai_bind_addr),
        &topology_path,
        &stage0_log,
        case.native_mtp_enabled,
        case.batched_verify_enabled,
    )?;
    wait_openai_ready(
        client,
        case.openai_bind_addr,
        args.server.startup_timeout_secs,
    )
    .context("stage 0 OpenAI server did not become ready")?;

    let response = client
        .post(format!(
            "http://{}/v1/chat/completions",
            case.openai_bind_addr
        ))
        .json(&json!({
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": args.runtime.prompt,
                },
            ],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }))
        .send()
        .context("failed to send OpenAI chat completion request")?;
    let http_status = response.status().as_u16();
    let body: Value = response.json().context("failed to parse OpenAI response")?;
    let content = body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let completion_tokens = body
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64);

    drop(_stage0);
    stage1.stop_and_collect(&stage1_log)?;

    let metrics = read_metrics(&stage0_log, &stage1_log)?;
    Ok(NativeMtpOpenAiCaseReport {
        case: case.case,
        native_mtp_enabled: case.native_mtp_enabled,
        batched_verify_enabled: case.batched_verify_enabled,
        http_status,
        content,
        completion_tokens,
        openai_bind_addr: case.openai_bind_addr.to_string(),
        stage0_bind_addr: case.stage0_bind_addr.to_string(),
        stage0_endpoint_addr: case.stage0_endpoint_addr.to_string(),
        stage1_bind_addr: case.stage1_bind_addr.to_string(),
        stage1_endpoint_addr: case.stage1_endpoint_addr.to_string(),
        stage0_config: stage0_config_path.display().to_string(),
        stage1_config: stage1_config_path.display().to_string(),
        topology_config: topology_path.display().to_string(),
        stage0_log: stage0_log.display().to_string(),
        stage1_log: stage1_log.display().to_string(),
        stage1_launch_mode: stage1_launch.launch_mode.to_string(),
        stage1_remote_config: stage1_launch.remote_config,
        stage1_remote_topology: stage1_launch.remote_topology,
        stage1_remote_log: stage1_launch.remote_log,
        metrics,
    })
}

fn stage_config_json(args: &NativeMtpOpenAiAbArgs, stage: OpenAiStageConfig<'_>) -> Value {
    json!({
        "run_id": stage.run_id,
        "topology_id": "native-mtp-open-ai-ab",
        "model_id": stage.model_id,
        "model_path": stage.model_path,
        "stage_id": stage.stage_id,
        "stage_index": stage.stage_index,
        "layer_start": stage.layer_start,
        "layer_end": stage.layer_end,
        "ctx_size": args.runtime.ctx_size,
        "lane_count": 1,
        "n_batch": args.runtime.n_batch,
        "n_ubatch": args.runtime.n_ubatch,
        "n_gpu_layers": args.runtime.n_gpu_layers,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "flash_attn_type": protocol_flash_attn(args.runtime.flash_attn),
        "filter_tensors_on_load": true,
        "load_mode": "runtime-slice",
        "bind_addr": stage.bind_addr,
        "upstream": stage.upstream,
        "downstream": stage.downstream,
    })
}

fn start_stage1(
    args: &NativeMtpOpenAiAbArgs,
    run_id: &str,
    config_path: &Path,
    topology_path: &Path,
    log_path: &Path,
    native_mtp_enabled: bool,
    batched_verify_enabled: bool,
) -> Result<(StageHandle, Stage1LaunchReport)> {
    if args.external_stage1 {
        fs::write(log_path, "stage 1 was externally managed by the caller\n")
            .with_context(|| format!("failed to create {}", log_path.display()))?;
        return Ok((
            StageHandle::External,
            Stage1LaunchReport {
                launch_mode: "external",
                remote_config: None,
                remote_topology: None,
                remote_log: None,
            },
        ));
    }

    if let Some(host) = args.stage1_ssh_host.as_deref() {
        let (remote, report) = spawn_remote_stage1(
            args,
            host,
            run_id,
            config_path,
            topology_path,
            native_mtp_enabled,
            batched_verify_enabled,
        )?;
        return Ok((StageHandle::Remote(remote), report));
    }

    let local = spawn_stage(
        args,
        config_path,
        None,
        topology_path,
        log_path,
        native_mtp_enabled,
        batched_verify_enabled,
    )?;
    Ok((
        StageHandle::Local(local),
        Stage1LaunchReport {
            launch_mode: "local",
            remote_config: None,
            remote_topology: None,
            remote_log: None,
        },
    ))
}

fn spawn_stage(
    args: &NativeMtpOpenAiAbArgs,
    config_path: &Path,
    openai_bind_addr: Option<SocketAddr>,
    topology_path: &Path,
    log_path: &Path,
    native_mtp_enabled: bool,
    batched_verify_enabled: bool,
) -> Result<ChildGuard> {
    let log = File::create(log_path)
        .with_context(|| format!("failed to create {}", log_path.display()))?;
    let mut command = Command::new(&args.server.stage_server_bin);
    command.args([
        "serve-binary",
        "--config",
        config_path
            .to_str()
            .context("stage config path is not valid UTF-8")?,
        "--topology",
        topology_path
            .to_str()
            .context("topology path is not valid UTF-8")?,
        "--activation-width",
        &args.activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
        "--telemetry-level",
        "debug",
    ]);
    if let Some(openai_bind_addr) = openai_bind_addr {
        command.args(["--openai-bind-addr", &openai_bind_addr.to_string()]);
    }
    command.env("SKIPPY_TELEMETRY_STDERR", "1");
    if native_mtp_enabled {
        command.env(NATIVE_MTP_ENABLED_ENV, "1");
    } else {
        command.env(NATIVE_MTP_ENABLED_ENV, "0");
    }
    if batched_verify_enabled {
        command.env_remove(BATCHED_VERIFY_ENV);
    } else {
        command.env(BATCHED_VERIFY_ENV, "0");
    }
    command.stdout(Stdio::from(log.try_clone()?));
    command.stderr(Stdio::from(log));
    ChildGuard::spawn(command)
}

fn spawn_remote_stage1(
    args: &NativeMtpOpenAiAbArgs,
    host: &str,
    run_id: &str,
    config_path: &Path,
    topology_path: &Path,
    native_mtp_enabled: bool,
    batched_verify_enabled: bool,
) -> Result<(RemoteStageGuard, Stage1LaunchReport)> {
    let remote_dir = format!(
        "{}/{}",
        args.stage1_remote_root.trim_end_matches('/'),
        run_id
    );
    let remote_config = format!("{remote_dir}/stage1.json");
    let remote_topology = format!("{remote_dir}/topology.json");
    let remote_log = format!("{remote_dir}/stage1.log");
    let remote_pid = format!("{remote_dir}/stage1.pid");
    let remote_bin = args
        .stage1_remote_stage_server_bin
        .as_deref()
        .context("missing stage 1 remote binary")?;

    ssh_success(host, &format!("mkdir -p {}", shell_quote(&remote_dir)))
        .with_context(|| format!("create remote stage directory on {host}"))?;
    scp_to(host, config_path, &remote_config).with_context(|| {
        format!(
            "copy stage 1 config {} to {host}:{remote_config}",
            config_path.display()
        )
    })?;
    scp_to(host, topology_path, &remote_topology).with_context(|| {
        format!(
            "copy topology {} to {host}:{remote_topology}",
            topology_path.display()
        )
    })?;

    let command = remote_stage_command(RemoteStageCommand {
        workdir: args.stage1_remote_workdir.as_deref(),
        remote_bin,
        remote_config: &remote_config,
        remote_topology: &remote_topology,
        remote_log: &remote_log,
        remote_pid: &remote_pid,
        activation_width: args.activation_width,
        activation_wire_dtype: &args.activation_wire_dtype,
        native_mtp_enabled,
        batched_verify_enabled,
    });
    let output = std::process::Command::new("ssh")
        .arg(host)
        .arg(command)
        .output()
        .with_context(|| format!("start remote stage 1 on {host}"))?;
    if !output.status.success() {
        bail!(
            "remote stage 1 start on {host} failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let pid = stdout
        .lines()
        .rev()
        .find_map(|line| line.trim().parse::<u32>().ok())
        .with_context(|| format!("remote stage 1 on {host} did not print a pid"))?;

    Ok((
        RemoteStageGuard {
            host: host.to_string(),
            pid: Some(pid),
            remote_log: remote_log.clone(),
        },
        Stage1LaunchReport {
            launch_mode: "ssh",
            remote_config: Some(remote_config),
            remote_topology: Some(remote_topology),
            remote_log: Some(remote_log),
        },
    ))
}

struct RemoteStageCommand<'a> {
    workdir: Option<&'a str>,
    remote_bin: &'a str,
    remote_config: &'a str,
    remote_topology: &'a str,
    remote_log: &'a str,
    remote_pid: &'a str,
    activation_width: i32,
    activation_wire_dtype: &'a str,
    native_mtp_enabled: bool,
    batched_verify_enabled: bool,
}

fn remote_stage_command(args: RemoteStageCommand<'_>) -> String {
    let mut command = String::new();
    if let Some(workdir) = args.workdir {
        command.push_str("cd ");
        command.push_str(&shell_quote(workdir));
        command.push_str(" && ");
    }
    command.push_str("SKIPPY_TELEMETRY_STDERR=1 ");
    command.push_str("SKIPPY_NATIVE_MTP_ENABLED=");
    command.push_str(if args.native_mtp_enabled { "1 " } else { "0 " });
    if !args.batched_verify_enabled {
        command.push_str("SKIPPY_NATIVE_MTP_BATCHED_VERIFY=0 ");
    }
    command.push_str("nohup ");
    command.push_str(&shell_quote(args.remote_bin));
    command.push_str(" serve-binary --config ");
    command.push_str(&shell_quote(args.remote_config));
    command.push_str(" --topology ");
    command.push_str(&shell_quote(args.remote_topology));
    command.push_str(" --activation-width ");
    command.push_str(&args.activation_width.to_string());
    command.push_str(" --activation-wire-dtype ");
    command.push_str(&shell_quote(args.activation_wire_dtype));
    command.push_str(" --telemetry-level debug > ");
    command.push_str(&shell_quote(args.remote_log));
    command.push_str(" 2>&1 < /dev/null & pid=$!; echo $pid > ");
    command.push_str(&shell_quote(args.remote_pid));
    command.push_str("; echo $pid");
    command
}

impl StageHandle {
    fn stop_and_collect(self, stage1_log: &Path) -> Result<()> {
        match self {
            StageHandle::Local(guard) => {
                drop(guard);
                Ok(())
            }
            StageHandle::External => Ok(()),
            StageHandle::Remote(mut guard) => guard.stop_and_collect(stage1_log),
        }
    }
}

impl RemoteStageGuard {
    fn stop_and_collect(&mut self, local_log: &Path) -> Result<()> {
        self.terminate();
        match scp_from(&self.host, &self.remote_log, local_log) {
            Ok(()) => Ok(()),
            Err(error) => {
                let note = format!("failed to collect remote stage 1 log: {error:#}\n");
                fs::write(local_log, note)
                    .with_context(|| format!("failed to write {}", local_log.display()))
            }
        }
    }

    fn terminate(&mut self) {
        let Some(pid) = self.pid.take() else {
            return;
        };
        let command = format!("kill {pid} 2>/dev/null || true; sleep 0.2");
        let _ = std::process::Command::new("ssh")
            .arg(&self.host)
            .arg(command)
            .status();
    }
}

impl Drop for RemoteStageGuard {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn ssh_success(host: &str, remote_command: &str) -> Result<()> {
    let status = Command::new("ssh")
        .arg(host)
        .arg(remote_command)
        .status()
        .with_context(|| format!("run ssh command on {host}"))?;
    if !status.success() {
        bail!("ssh command on {host} failed with status {status}");
    }
    Ok(())
}

fn scp_to(host: &str, local_path: &Path, remote_path: &str) -> Result<()> {
    let status = Command::new("scp")
        .arg(local_path)
        .arg(format!("{host}:{remote_path}"))
        .status()
        .with_context(|| format!("copy {} to {host}:{remote_path}", local_path.display()))?;
    if !status.success() {
        bail!("scp to {host}:{remote_path} failed with status {status}");
    }
    Ok(())
}

fn scp_from(host: &str, remote_path: &str, local_path: &Path) -> Result<()> {
    let status = Command::new("scp")
        .arg(format!("{host}:{remote_path}"))
        .arg(local_path)
        .status()
        .with_context(|| format!("copy {host}:{remote_path} to {}", local_path.display()))?;
    if !status.success() {
        bail!("scp from {host}:{remote_path} failed with status {status}");
    }
    Ok(())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn wait_openai_ready(client: &Client, addr: SocketAddr, timeout_secs: u64) -> Result<()> {
    let attempts = timeout_secs.saturating_mul(4).max(1);
    let url = format!("http://{addr}/v1/models");
    let mut last_error = None;
    for _ in 0..attempts {
        match client.get(&url).send() {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(response) => last_error = Some(format!("HTTP {}", response.status())),
            Err(error) => last_error = Some(error.to_string()),
        }
        thread::sleep(Duration::from_millis(250));
    }
    bail!(
        "timed out waiting for {url}: {}",
        last_error.unwrap_or_else(|| "no attempts made".to_string())
    );
}

fn read_metrics(stage0_log: &Path, stage1_log: &Path) -> Result<NativeMtpOpenAiMetricsReport> {
    let mut metrics = NativeMtpOpenAiMetricsReport::default();
    for path in [stage0_log, stage1_log] {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        metrics.fatal_error_events += count_fatal_log_lines(&text);
        for line in text.lines() {
            let Ok(event) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            apply_telemetry_event(&mut metrics, &event);
        }
    }
    Ok(metrics)
}

fn apply_telemetry_event(metrics: &mut NativeMtpOpenAiMetricsReport, event: &Value) {
    let Some(name) = event.get("event").and_then(Value::as_str) else {
        return;
    };
    let attrs = event.get("attributes").unwrap_or(&Value::Null);
    match name {
        "stage.openai_decode_token" => {
            metrics.decode_token_events += 1;
            apply_native_mtp_verification(metrics, attrs);
        }
        "stage.openai_native_mtp_verify" => {
            metrics.batched_verify_events += 1;
            match apply_native_mtp_verification(metrics, attrs) {
                Some("accepted") => metrics.batched_accepted_events += 1,
                Some("rejected") => metrics.batched_rejected_events += 1,
                _ => {}
            }
        }
        "stage.openai_decode" | "stage.openai_generation_summary" => {
            apply_generation_summary(metrics, attrs);
        }
        _ => {}
    }
}

fn apply_generation_summary(metrics: &mut NativeMtpOpenAiMetricsReport, attrs: &Value) {
    metrics.native_mtp_enabled |= attr_bool(attrs, "llama_stage.native_mtp.enabled");
    if let Some(drafted) = attr_u64(attrs, "llama_stage.native_mtp.drafted") {
        metrics.drafted_tokens = drafted;
    }
    if let Some(accepted) = attr_u64(attrs, "llama_stage.native_mtp.accepted") {
        metrics.accepted_tokens = accepted;
    }
    if let Some(rejected) = attr_u64(attrs, "llama_stage.native_mtp.rejected") {
        metrics.rejected_tokens = rejected;
    }
    if let Some(pending) = attr_u64(attrs, "llama_stage.native_mtp.pending") {
        metrics.pending_tokens = pending;
    }
    if let Some(verifications) = attr_u64(attrs, "llama_stage.native_mtp.verifications") {
        metrics.verification_count = verifications;
    }
    if let Some(proposal_compute_us) = attr_i64(attrs, "llama_stage.native_mtp.proposal_compute_us")
    {
        metrics.proposal_compute_us = proposal_compute_us;
    }
    if let Some(verification_compute_us) =
        attr_i64(attrs, "llama_stage.native_mtp.verification_compute_us")
    {
        metrics.verification_compute_us = verification_compute_us;
    }
    if let Some(accept_rate) = attrs
        .get("llama_stage.native_mtp.accept_rate")
        .and_then(Value::as_f64)
    {
        metrics.accept_rate = accept_rate;
    }
}

fn apply_native_mtp_verification<'a>(
    metrics: &mut NativeMtpOpenAiMetricsReport,
    attrs: &'a Value,
) -> Option<&'a str> {
    let verification = attrs
        .get("llama_stage.native_mtp.verification")
        .and_then(Value::as_str)?;
    match verification {
        "accepted" => {
            metrics.native_mtp_enabled = true;
            metrics.accepted_tokens += 1;
            metrics.verification_count += 1;
        }
        "rejected" => {
            metrics.native_mtp_enabled = true;
            metrics.rejected_tokens += 1;
            metrics.verification_count += 1;
        }
        _ => {}
    }
    Some(verification)
}

fn count_fatal_log_lines(text: &str) -> u64 {
    text.lines()
        .filter(|line| {
            line.contains("panicked")
                || line.contains("service_unavailable")
                || line.contains("llama_decode failed for MTP sidecar sync")
        })
        .count() as u64
}

fn attr_bool(attrs: &Value, key: &str) -> bool {
    attrs.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn attr_u64(attrs: &Value, key: &str) -> Option<u64> {
    attrs.get(key).and_then(Value::as_u64)
}

fn attr_i64(attrs: &Value, key: &str) -> Option<i64> {
    attrs.get(key).and_then(Value::as_i64)
}

fn case_addr(addr: SocketAddr, port_offset: u16, case_index: u16) -> Result<SocketAddr> {
    let offset = port_offset
        .checked_mul(case_index)
        .context("case port offset exceeds u16")?;
    offset_port(addr, offset)
}

fn offset_port(mut addr: SocketAddr, offset: u16) -> Result<SocketAddr> {
    let port = addr
        .port()
        .checked_add(offset)
        .context("batched port offset exceeds u16")?;
    addr.set_port(port);
    Ok(addr)
}

fn protocol_flash_attn(value: crate::cli::FlashAttentionArg) -> &'static str {
    match value {
        crate::cli::FlashAttentionArg::Auto => "auto",
        crate::cli::FlashAttentionArg::Disabled => "disabled",
        crate::cli::FlashAttentionArg::Enabled => "enabled",
    }
}

fn write_stage_config(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn emit_report<T: serde::Serialize>(report: &T, report_out: Option<&Path>) -> Result<()> {
    let json = serde_json::to_vec_pretty(report)?;
    if let Some(path) = report_out {
        fs::write(path, &json).with_context(|| format!("failed to write {}", path.display()))?;
    }
    println!("{}", String::from_utf8(json)?);
    Ok(())
}

fn status(matches: bool) -> &'static str {
    if matches { "pass" } else { "fail" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_parser_extracts_native_mtp_counts() {
        let mut metrics = NativeMtpOpenAiMetricsReport::default();
        apply_telemetry_event(
            &mut metrics,
            &json!({
                "event": "stage.openai_native_mtp_verify",
                "attributes": {
                    "llama_stage.native_mtp.verification": "accepted",
                }
            }),
        );
        apply_telemetry_event(
            &mut metrics,
            &json!({
                "event": "stage.openai_decode_token",
                "attributes": {}
            }),
        );
        apply_telemetry_event(
            &mut metrics,
            &json!({
                "event": "stage.openai_decode",
                "attributes": {
                    "llama_stage.native_mtp.enabled": true,
                    "llama_stage.native_mtp.drafted": 4,
                    "llama_stage.native_mtp.accepted": 3,
                    "llama_stage.native_mtp.rejected": 1,
                    "llama_stage.native_mtp.pending": 0,
                    "llama_stage.native_mtp.verifications": 4,
                    "llama_stage.native_mtp.accept_rate": 0.75,
                    "llama_stage.native_mtp.proposal_compute_us": 11,
                    "llama_stage.native_mtp.verification_compute_us": 22,
                }
            }),
        );

        assert!(metrics.native_mtp_enabled);
        assert_eq!(metrics.drafted_tokens, 4);
        assert_eq!(metrics.accepted_tokens, 3);
        assert_eq!(metrics.rejected_tokens, 1);
        assert_eq!(metrics.verification_count, 4);
        assert_eq!(metrics.accept_rate, 0.75);
        assert_eq!(metrics.proposal_compute_us, 11);
        assert_eq!(metrics.verification_compute_us, 22);
        assert_eq!(metrics.batched_verify_events, 1);
        assert_eq!(metrics.batched_accepted_events, 1);
        assert_eq!(metrics.decode_token_events, 1);
    }

    #[test]
    fn fatal_counter_ignores_connection_retry_noise() {
        let text = "\
downstream connect retry: error=Connection refused (os error 61)
llama_decode failed for MTP sidecar sync
";
        assert_eq!(count_fatal_log_lines(text), 1);
    }

    #[test]
    fn remote_stage_command_quotes_paths_and_sets_mtp_env() {
        let command = remote_stage_command(RemoteStageCommand {
            workdir: Some("/tmp/work dir"),
            remote_bin: "target/debug/skippy-server",
            remote_config: "/tmp/run/stage'1.json",
            remote_topology: "/tmp/run/topology.json",
            remote_log: "/tmp/run/stage1.log",
            remote_pid: "/tmp/run/stage1.pid",
            activation_width: 2048,
            activation_wire_dtype: "f16",
            native_mtp_enabled: true,
            batched_verify_enabled: false,
        });

        assert!(command.contains("cd '/tmp/work dir' && "));
        assert!(command.contains("SKIPPY_NATIVE_MTP_ENABLED=1"));
        assert!(command.contains("SKIPPY_NATIVE_MTP_BATCHED_VERIFY=0"));
        assert!(command.contains("'target/debug/skippy-server' serve-binary"));
        assert!(command.contains("'/tmp/run/stage'\"'\"'1.json'"));

        let batched_command = remote_stage_command(RemoteStageCommand {
            batched_verify_enabled: true,
            ..RemoteStageCommand {
                workdir: None,
                remote_bin: "/bin/skippy-server",
                remote_config: "/tmp/stage1.json",
                remote_topology: "/tmp/topology.json",
                remote_log: "/tmp/stage1.log",
                remote_pid: "/tmp/stage1.pid",
                activation_width: 2048,
                activation_wire_dtype: "f16",
                native_mtp_enabled: true,
                batched_verify_enabled: false,
            }
        });
        assert!(!batched_command.contains("SKIPPY_NATIVE_MTP_BATCHED_VERIFY"));
    }
}
