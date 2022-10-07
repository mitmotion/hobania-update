use std::{
    error::Error,
    io::Cursor,
    net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4},
    time::{Duration, Instant},
};

use tokio::{net::UdpSocket, time::timeout};
use veloren_server::query_server::{Pong, QueryServerPacketKind, VELOREN_HEADER};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let dest = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 14006);

    // Ping
    let (result, elapsed) = send_query(
        QueryServerPacketKind::Ping(veloren_server::query_server::Ping),
        dest,
    )
    .await?;

    if matches!(result, QueryServerPacketKind::Pong(Pong)) {
        println!("Ping response took {}ms", elapsed.as_millis());
    } else {
        return Err("Unexpected response to Ping received".into());
    }

    // ServerInfoQuery
    let (result, elapsed) = send_query(
        QueryServerPacketKind::ServerInfoQuery(veloren_server::query_server::ServerInfoQuery {
            padding: [0xFF; 512],
        }),
        dest,
    )
    .await?;

    if let QueryServerPacketKind::ServerInfoResponse(_) = result {
        println!(
            "ServerInfoQuery response took {}ms - data: {:?}",
            elapsed.as_millis(),
            result
        );
    } else {
        return Err("Unexpected response to ServerInfoQuery received".into());
    }

    Ok(())
}

async fn send_query(
    query: QueryServerPacketKind,
    dest: SocketAddr,
) -> Result<(QueryServerPacketKind, Duration), Box<dyn Error>> {
    let socket = UdpSocket::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0)).await?;

    let mut pipeline = protocol::wire::dgram::Pipeline::new(
        protocol::wire::middleware::pipeline::default(),
        protocol::Settings::default(),
    );

    let mut buf = Vec::<u8>::new();
    // All outgoing datagrams must start with the "VELOREN" header
    buf.append(&mut VELOREN_HEADER.to_vec());

    let mut cursor = Cursor::new(&mut buf);
    cursor.set_position(VELOREN_HEADER.len() as u64);
    pipeline.send_to(&mut cursor, &query)?;

    let query_sent = Instant::now();
    socket.send_to(buf.as_slice(), dest).await?;

    let mut buf = vec![0; 1500];
    let _ = timeout(Duration::from_secs(2), socket.recv_from(&mut buf))
        .await
        .expect("Socket receive failed")
        .expect("Socket receive timeout");

    let packet: QueryServerPacketKind = pipeline.receive_from(&mut Cursor::new(&mut buf))?;

    Ok((packet, query_sent.elapsed()))
}
