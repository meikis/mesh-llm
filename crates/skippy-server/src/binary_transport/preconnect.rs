use super::*;

const WARM_DOWNSTREAM_RETRY_SLEEP: Duration = Duration::from_millis(500);
const WARM_DOWNSTREAM_SLOT_POLL: Duration = Duration::from_millis(50);
const WARM_DOWNSTREAM_CONNECT_TIMEOUT_SECS: u64 = 2;

pub(super) fn try_initial_downstream_preconnect(
    config: &StageConfig,
    warm_downstream: &Arc<Mutex<Option<TcpStream>>>,
    downstream_connect_timeout_secs: u64,
) {
    if config.downstream.is_none() {
        return;
    }
    match connect_binary_downstream(config, downstream_connect_timeout_secs) {
        Ok(Some(stream)) => {
            let local_addr = stream.local_addr().ok();
            let peer_addr = stream.peer_addr().ok();
            eprintln!(
                "downstream initial preconnect ready: stage_id={} local={local_addr:?} remote={peer_addr:?}",
                config.stage_id,
            );
            store_warm_stream(warm_downstream, stream);
        }
        Ok(None) => {}
        Err(error) => {
            eprintln!(
                "downstream initial preconnect failed: stage_id={} error={error:#}",
                config.stage_id,
            );
        }
    }
}

pub(super) fn spawn_downstream_preconnector(
    config: StageConfig,
    warm_downstream: Arc<Mutex<Option<TcpStream>>>,
    shutdown: Arc<AtomicBool>,
    downstream_connect_timeout_secs: u64,
) {
    if config.downstream.is_none() {
        return;
    }
    let thread_name = format!("skippy-warm-downstream-{}", config.stage_index);
    let _ = thread::Builder::new().name(thread_name).spawn(move || {
        run_downstream_preconnector(
            config,
            warm_downstream,
            shutdown,
            downstream_connect_timeout_secs,
        );
    });
}

fn run_downstream_preconnector(
    config: StageConfig,
    warm_downstream: Arc<Mutex<Option<TcpStream>>>,
    shutdown: Arc<AtomicBool>,
    downstream_connect_timeout_secs: u64,
) {
    let connect_timeout_secs =
        downstream_connect_timeout_secs.clamp(1, WARM_DOWNSTREAM_CONNECT_TIMEOUT_SECS);
    while !shutdown.load(Ordering::SeqCst) {
        if warm_slot_is_full(&warm_downstream) {
            thread::sleep(WARM_DOWNSTREAM_SLOT_POLL);
            continue;
        }
        match connect_binary_downstream(&config, connect_timeout_secs) {
            Ok(Some(stream)) => {
                let local_addr = stream.local_addr().ok();
                let peer_addr = stream.peer_addr().ok();
                eprintln!(
                    "downstream warm preconnect ready: stage_id={} local={local_addr:?} remote={peer_addr:?}",
                    config.stage_id,
                );
                store_warm_stream(&warm_downstream, stream);
            }
            Ok(None) => return,
            Err(error) => {
                eprintln!(
                    "downstream warm preconnect failed: stage_id={} error={error:#}",
                    config.stage_id,
                );
                thread::sleep(WARM_DOWNSTREAM_RETRY_SLEEP);
            }
        }
    }
}

fn warm_slot_is_full(warm_downstream: &Arc<Mutex<Option<TcpStream>>>) -> bool {
    warm_downstream
        .lock()
        .map(|guard| guard.is_some())
        .unwrap_or(true)
}

fn store_warm_stream(warm_downstream: &Arc<Mutex<Option<TcpStream>>>, stream: TcpStream) {
    let Ok(mut guard) = warm_downstream.lock() else {
        return;
    };
    if guard.is_none() {
        *guard = Some(stream);
    }
}
