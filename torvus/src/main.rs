use client::{Client as VelorenClient, Event as VelorenEvent};
use common::{clock::Clock, comp, util::DISPLAY_VERSION_LONG};
use crossbeam_channel::{Receiver, Sender};
use serenity::{
    async_trait,
    model::{
        channel::Message,
        event::ResumedEvent,
        gateway::{Activity, Ready},
        id::{ChannelId, GuildId},
        user::OnlineStatus,
    },
    prelude::*,
    utils::Colour,
};
use std::{
    env,
    net::{SocketAddr, ToSocketAddrs},
    thread,
    time::Duration,
};

const TPS: u64 = 60;

enum VelorenMessage {
    Connect,
    Disconnect,
    Chat(String),
}

enum DiscordMessage {
    Chat { nickname: String, msg: Message },
}

struct Handler {
    messages_rx: Receiver<VelorenMessage>,
    server_addr: SocketAddr,
    guild: u64,
    bridge_channel: u64,
    discord_msgs_tx: Sender<DiscordMessage>,
}

impl Handler {
    pub fn new<A: Into<SocketAddr> + Clone>(
        addr: A,
        guild: u64,
        bridge_channel: u64,
        veloren_username: String,
        veloren_password: String,
        trusted_auth_server: String,
    ) -> Self {
        let (messages_tx, messages_rx) = crossbeam_channel::unbounded();
        let (discord_msgs_tx, discord_msgs_rx) = crossbeam_channel::unbounded();

        let socket = addr.clone().into();
        tokio::task::spawn_blocking(move || {
            let mut retry_cnt = [0u32; 3];

            'connect: loop {
                log::debug!("Connecting...");
                let mut veloren_client = match VelorenClient::new(socket, None) {
                    Ok(client) => client,
                    Err(e) => {
                        log::error!(
                            "Failed to connect to Veloren server: {:?}, retry: {}",
                            e,
                            retry_cnt[0]
                        );
                        retry_cnt[0] += 1;
                        thread::sleep(Duration::from_millis(500) * retry_cnt[0]);
                        continue 'connect;
                    },
                };
                retry_cnt[0] = 0;

                if let Err(e) = veloren_client.register(
                    veloren_username.clone(),
                    veloren_password.clone(),
                    |auth_server| auth_server == trusted_auth_server,
                ) {
                    log::error!(
                        "Failed to switch to registered state: {:?}, retry: {}",
                        e,
                        retry_cnt[1]
                    );
                    retry_cnt[1] += 1;
                    thread::sleep(Duration::from_secs(30) * retry_cnt[1]);
                    continue 'connect;
                }
                retry_cnt[1] = 0;

                messages_tx.send(VelorenMessage::Connect).unwrap();

                log::debug!("Logged in.");
                let mut clock = Clock::new(Duration::from_secs_f64(1.0 / TPS as f64));
                loop {
                    let events = match veloren_client.tick(
                        comp::ControllerInputs::default(),
                        clock.dt(),
                        |_| {},
                    ) {
                        Ok(events) => events,
                        Err(e) => {
                            log::error!("Failed to tick client: {:?}, retry: {}", e, retry_cnt[2]);
                            retry_cnt[2] += 1;
                            thread::sleep(Duration::from_secs(10) * retry_cnt[2]);
                            messages_tx.send(VelorenMessage::Disconnect).unwrap();
                            continue 'connect;
                        },
                    };
                    retry_cnt[2] = 0;

                    if let Ok(DiscordMessage::Chat { nickname, msg }) = discord_msgs_rx.try_recv() {
                        let msg: Message = msg;
                        veloren_client.send_chat(format!("/alias D_{}", nickname));
                        veloren_client.send_chat(deunicode::deunicode(&msg.content).to_string());
                    }

                    for event in events {
                        match event {
                            VelorenEvent::Chat(msg) => messages_tx
                                .send(VelorenMessage::Chat(
                                    veloren_client.format_message(&msg, true),
                                ))
                                .unwrap(),
                            VelorenEvent::Disconnect => {
                                messages_tx.send(VelorenMessage::Disconnect).unwrap()
                            },
                            VelorenEvent::DisconnectionNotification(_) => {
                                log::debug!("Will be disconnected soon! :/")
                            },
                            VelorenEvent::Notification(notification) => {
                                log::debug!("Notification: {:?}", notification);
                            },
                            _ => {},
                        }
                    }
                    veloren_client.cleanup();

                    clock.tick();
                }
            }
        });

        Self {
            messages_rx,
            server_addr: addr.into(),
            guild,
            bridge_channel,
            discord_msgs_tx,
        }
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        log::info!("Connected as {}", ready.user.name);

        ctx.set_presence(Some(Activity::playing("Veloren")), OnlineStatus::Online)
            .await;

        let channel_id = ChannelId(self.bridge_channel);
        let server_addr = self.server_addr;
        let messages_rx = self.messages_rx.clone();

        let _handle = tokio::spawn(async move {
            loop {
                let msg = match messages_rx.try_recv() {
                    Ok(msg) => msg,
                    Err(crossbeam_channel::TryRecvError::Empty) => {
                        thread::sleep(Duration::from_millis(50));
                        continue;
                    },
                    _ => break,
                };

                let send_msg = |text| {
                    channel_id.send_message(&ctx.http, |m| {
                        m.embed(|e| e.colour(Colour::BLURPLE).title(text))
                    })
                };

                match msg {
                    VelorenMessage::Chat(chat_msg) => {
                        if !chat_msg.starts_with("[You]") {
                            log::debug!("Veloren -> {}", chat_msg);
                            if let Err(e) = send_msg(chat_msg).await {
                                log::error!("Failed to send discord message: {}", e);
                            }
                        }
                    },
                    VelorenMessage::Connect => {
                        ctx.online().await;
                        log::info!("Server bridge connected to {}.", server_addr);
                    },
                    VelorenMessage::Disconnect => {
                        ctx.invisible().await;
                        log::info!("Server bridge disconnected from {}.", server_addr);
                    },
                }
            }
        });
    }

    async fn resume(&self, _: Context, _: ResumedEvent) {
        log::info!("Connection to discord resumed.");
    }

    async fn message(&self, ctx: Context, msg: Message) {
        let guild = GuildId(self.guild);

        if *msg.channel_id.as_u64() != self.bridge_channel {
            return;
        }

        if !msg.author.bot {
            log::debug!("Discord -> {}", &deunicode::deunicode(&msg.content));
            self.discord_msgs_tx
                .send(DiscordMessage::Chat {
                    nickname: msg
                        .author
                        .nick_in(ctx.http, guild)
                        .await
                        .unwrap_or(msg.clone().author.name),
                    msg,
                })
                .unwrap();
        }
    }
}

#[tokio::main]
async fn main() {
    use env_logger::Env;

    kankyo::init().ok();
    env_logger::Builder::from_env(Env::default().default_filter_or("warn,veloren_torvus=debug"))
        .init();

    let token: String = env_key("DISCORD_TOKEN");
    let veloren_server: SocketAddr = env::var("VELOREN_SERVER")
        .expect("No environment variable 'VELOREN_SERVER' found.")
        .to_socket_addrs()
        .expect("Invalid address!")
        .next()
        .expect("Invalid address!");
    let guild = env_key("DISCORD_GUILD");
    let bridge_channel = env_key("DISCORD_CHANNEL");

    let veloren_username = env_key("VELOREN_USERNAME");
    let veloren_password = env_key("VELOREN_PASSWORD");
    let trusted_auth_server = env_key("VELOREN_TRUSTED_AUTH_SERVER");

    let handler = Handler::new(
        veloren_server,
        guild,
        bridge_channel,
        veloren_username,
        veloren_password,
        trusted_auth_server,
    );

    log::info!("Veloren-Common/Client version: {}", *DISPLAY_VERSION_LONG);

    let mut client = Client::builder(&token)
        .event_handler(handler)
        .await
        .expect("Failed to create serenity client!");

    client.start().await.expect("Failed to start client.");
}

fn env_key<T>(key: &str) -> T
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    env::var(key)
        .unwrap_or_else(|_| panic!("No environment variable '{}' found.", key))
        .parse()
        .unwrap_or_else(|_| panic!("'{}' couldn't be parsed.", key.to_string()))
}
