use anyhow::{Context, Result};

pub(super) fn socket_addr_http_url(addr: std::net::SocketAddr) -> String {
    format!("http://{addr}")
}

pub(super) fn listener_http_url(
    listener: &tokio::net::TcpListener,
    fallback_port: u16,
    label: &str,
) -> String {
    listener_http_endpoint(listener, fallback_port, label).0
}

pub(super) fn listener_http_endpoint(
    listener: &tokio::net::TcpListener,
    fallback_port: u16,
    label: &str,
) -> (String, u16) {
    listener
        .local_addr()
        .map(|addr| (socket_addr_http_url(addr), addr.port()))
        .unwrap_or_else(|err| {
            tracing::warn!("{label}: failed to read listener address: {err}");
            (format!("http://localhost:{fallback_port}"), fallback_port)
        })
}

pub(super) async fn bind_runtime_tcp_listener(
    port: u16,
    listen_all: bool,
    label: &str,
) -> Result<tokio::net::TcpListener> {
    let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
    tokio::net::TcpListener::bind(format!("{addr}:{port}"))
        .await
        .with_context(|| format!("Failed to bind {label} to port {port}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn listener_http_url_uses_bound_ephemeral_addr() {
        let listener = bind_runtime_tcp_listener(0, false, "test listener")
            .await
            .expect("ephemeral listener should bind");
        let addr = listener
            .local_addr()
            .expect("bound listener should expose local address");

        let url = listener_http_url(&listener, 0, "test listener");

        assert_eq!(url, socket_addr_http_url(addr));
        assert_ne!(url, "http://localhost:0");
        assert!(!url.ends_with(":0"));
    }

    #[tokio::test]
    async fn bind_runtime_tcp_listener_uses_loopback_by_default() {
        let listener = bind_runtime_tcp_listener(0, false, "test listener")
            .await
            .expect("loopback listener should bind");
        let addr = listener
            .local_addr()
            .expect("bound listener should expose local address");

        assert_eq!(addr.ip(), std::net::Ipv4Addr::LOCALHOST);
        assert_ne!(addr.port(), 0);
    }

    #[tokio::test]
    async fn bind_runtime_tcp_listener_uses_unspecified_addr_when_listening_all() {
        let listener = bind_runtime_tcp_listener(0, true, "test listener")
            .await
            .expect("listen-all listener should bind");
        let addr = listener
            .local_addr()
            .expect("bound listener should expose local address");

        assert_eq!(addr.ip(), std::net::Ipv4Addr::UNSPECIFIED);
        assert_ne!(addr.port(), 0);
    }

    #[tokio::test]
    async fn listener_http_endpoint_reports_bound_url_and_port() {
        let listener = bind_runtime_tcp_listener(0, false, "test listener")
            .await
            .expect("ephemeral listener should bind");
        let addr = listener
            .local_addr()
            .expect("bound listener should expose local address");

        let (url, port) = listener_http_endpoint(&listener, 0, "test listener");

        assert_eq!(url, socket_addr_http_url(addr));
        assert_eq!(port, addr.port());
        assert_ne!(port, 0);
    }
}
