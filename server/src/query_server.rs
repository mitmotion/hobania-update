use crate::{settings::ServerBattleMode, ServerInfoRequest};
use common::resources::BattleMode;
use protocol::{wire::dgram, Protocol};
use std::{
    io,
    io::Cursor,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    time::Duration,
};
use tokio::{net::UdpSocket, sync::mpsc::UnboundedSender, time::timeout};
use tracing::{debug, info, trace};

// NOTE: Debug logging is disabled by default for this module - to enable it add
// veloren_server::query_server=trace to RUST_LOG

pub struct QueryServer {
    pub bind_addr: SocketAddr,
    pipeline: dgram::Pipeline<QueryServerPacketKind, protocol::wire::middleware::pipeline::Default>,
    server_info_request_sender: UnboundedSender<ServerInfoRequest>,
}

impl QueryServer {
    pub fn new(
        bind_addr: SocketAddr,
        server_info_request_sender: UnboundedSender<ServerInfoRequest>,
    ) -> Self {
        Self {
            bind_addr,
            pipeline: protocol::wire::dgram::Pipeline::new(
                protocol::wire::middleware::pipeline::default(),
                protocol::Settings::default(),
            ),
            server_info_request_sender,
        }
    }

    pub async fn run(&mut self) -> Result<(), io::Error> {
        let socket = UdpSocket::bind(self.bind_addr).await?;

        info!("Query Server running at {}", self.bind_addr);

        loop {
            let mut buf = vec![0; 1500];
            let (len, remote_addr) = socket.recv_from(&mut buf).await?;

            if !QueryServer::validate_datagram(len, &mut buf) {
                continue;
            }

            if let Err(e) = self.process_datagram(buf, remote_addr).await {
                debug!(?e, "Failed to process incoming datagram")
            }
        }
    }

    fn validate_datagram(len: usize, data: &mut Vec<u8>) -> bool {
        const VELOREN_HEADER: [u8; 7] = [0x56, 0x45, 0x4C, 0x4F, 0x52, 0x45, 0x4E];

        if len < 8 {
            trace!("Ignoring packet - too short");
            false
        } else if data[0..7] != VELOREN_HEADER {
            trace!("Ignoring packet - missing header");
            false
        } else {
            trace!("Validated packet, data: {:?}", data);

            // Discard the header after successful validation
            *data = data.split_off(7);
            true
        }
    }

    async fn process_datagram(
        &mut self,
        datagram: Vec<u8>,
        remote_addr: SocketAddr,
    ) -> Result<(), QueryError> {
        let packet: QueryServerPacketKind =
            self.pipeline.receive_from(&mut Cursor::new(datagram))?;

        debug!(?packet, ?remote_addr, "Query Server received packet");

        match packet {
            QueryServerPacketKind::Ping(_) => {
                QueryServer::send_response(remote_addr, &QueryServerPacketKind::Pong(Pong {}))
                    .await?
            },
            QueryServerPacketKind::ServerInfoQuery(ref _query) => {
                let (sender, receiver) = tokio::sync::oneshot::channel::<ServerInfoResponse>();
                let req = ServerInfoRequest {
                    response_sender: sender,
                };
                self.server_info_request_sender
                    .send(req)
                    .map_err(|e| QueryError::ChannelError(format!("{}", e)))?;

                tokio::spawn(async move {
                    match timeout(Duration::from_secs(2), async move {
                        match receiver.await {
                            Ok(response) => {
                                trace!(?response, "Sending ServerInfoResponse");
                                QueryServer::send_response(
                                    remote_addr,
                                    &QueryServerPacketKind::ServerInfoResponse(response),
                                )
                                .await
                                .expect("Failed to send response"); // TODO remove expect
                            },
                            Err(_) => {
                                //  Oneshot receive error
                            },
                        }
                    })
                    .await
                    {
                        Ok(_) => {},
                        Err(elapsed) => {
                            debug!(
                                ?elapsed,
                                "Timeout expired while waiting for ServerInfoResponse"
                            );
                        },
                    }
                });
            },
            QueryServerPacketKind::Pong(_) | QueryServerPacketKind::ServerInfoResponse(_) => {
                // Ignore any incoming packets
                debug!(?packet, "Dropping received response packet");
            },
        }

        Ok(())
    }

    async fn send_response(
        dest: SocketAddr,
        packet: &QueryServerPacketKind,
    ) -> Result<(), QueryError> {
        let socket =
            UdpSocket::bind(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0))).await?;

        let mut buf = Vec::<u8>::new();

        let mut pipeline = protocol::wire::dgram::Pipeline::new(
            protocol::wire::middleware::pipeline::default(),
            protocol::Settings::default(),
        );

        pipeline.send_to(&mut Cursor::new(&mut buf), packet)?;
        socket.send_to(buf.as_slice(), dest).await?;

        Ok(())
    }
}

#[derive(Debug)]
enum QueryError {
    NetworkError(io::Error),
    ProtocolError(protocol::Error),
    ChannelError(String),
}

impl From<protocol::Error> for QueryError {
    fn from(e: protocol::Error) -> Self { QueryError::ProtocolError(e) }
}

impl From<io::Error> for QueryError {
    fn from(e: io::Error) -> Self { QueryError::NetworkError(e) }
}

#[derive(protocol::Protocol, Clone, Debug, PartialEq)]
pub struct Ping;

#[derive(protocol::Protocol, Clone, Debug, PartialEq)]
pub struct Pong;

#[derive(protocol::Protocol, Clone, Debug, PartialEq)]
pub struct ServerInfoQuery;

#[derive(protocol::Protocol, Clone, Debug, PartialEq)]
#[protocol(discriminant = "integer")]
#[protocol(discriminator(u8))]
pub enum QueryServerPacketKind {
    #[protocol(discriminator(0x00))]
    Ping(Ping),
    #[protocol(discriminator(0x01))]
    Pong(Pong),
    #[protocol(discriminator(0xA0))]
    ServerInfoQuery(ServerInfoQuery),
    #[protocol(discriminator(0xA1))]
    ServerInfoResponse(ServerInfoResponse),
}

#[derive(Protocol, Debug, Clone, PartialEq)]
pub struct ServerInfoResponse {
    pub git_hash: String, /* TODO: use u8 array instead? String includes 8 bytes for capacity
                           * and length that we don't need */
    pub players_current: u16,
    pub players_max: u16,
    pub battle_mode: QueryBattleMode, // TODO: use a custom enum to avoid accidental breakage
}

#[derive(Protocol, Debug, Clone, PartialEq)]
#[protocol(discriminant = "integer")]
#[protocol(discriminator(u8))]
pub enum QueryBattleMode {
    #[protocol(discriminator(0x00))]
    GlobalPvP,
    #[protocol(discriminator(0x01))]
    GlobalPvE,
    #[protocol(discriminator(0x02))]
    PerPlayer,
}

impl From<ServerBattleMode> for QueryBattleMode {
    fn from(battle_mode: ServerBattleMode) -> Self {
        match battle_mode {
            ServerBattleMode::Global(x) => match x {
                BattleMode::PvP => QueryBattleMode::GlobalPvP,
                BattleMode::PvE => QueryBattleMode::GlobalPvE,
            },
            ServerBattleMode::PerPlayer { .. } => QueryBattleMode::PerPlayer,
        }
    }
}
