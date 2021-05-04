#![deny(unsafe_code)]

use common::clock::Clock;
use iced::{
    executor, image, scrollable, Align, Application, Command, Element, Image, Length, Scrollable,
    Settings, Subscription,
};
use server::{Event, Input, Server, ServerSettings};
use std::{
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
    time::Duration,
};
use tracing::{info, Level};
use tracing_subscriber::{filter::LevelFilter, EnvFilter, FmtSubscriber};

struct Gui {
    server: Arc<Mutex<Server>>,
    server_worker: JoinHandle<()>,
    scroll: scrollable::State,
    map_image: image::Handle,
}

#[derive(Debug, Clone)]
enum Message {}

impl Application for Gui {
    type Executor = executor::Default;
    type Flags = ();
    type Message = Message;

    fn new(_flags: Self::Flags) -> (Self, Command<Self::Message>) {
        // Create server
        let server =
            Server::new(ServerSettings::load()).expect("Failed to create server instance!");

        // Create image of map
        let map_image = {
            let sz = server.map().dimensions_lg.map(|e| 1 << e);
            let max_height = server.map().max_height;
            let bytes = server.map().rgba
                .iter()
                //.map(|px| (0..4).map(move |i| (((*px & ((1 << 15) - 1)) >> 2) as f32 * 255.0 / max_height) as u8))
                .map(|px| {
                    let rgba = px.to_le_bytes();
                    let b = [rgba[2], rgba[1], rgba[0], 0xFF];
                    (0..4).map(move |i| b[i])
                })
                .flatten()
                .collect::<Vec<_>>();
            image::Handle::from_pixels(sz.x, sz.y, bytes)
        };

        let server = Arc::new(Mutex::new(server));

        // Start server worker
        let server_worker = {
            let server = server.clone();
            thread::spawn(|| server_worker(server))
        };

        (
            Gui {
                server,
                server_worker,
                scroll: scrollable::State::default(),
                map_image,
            },
            Command::none(),
        )
    }

    fn title(&self) -> String { "Veloren Server".to_string() }

    fn update(&mut self, msg: Self::Message) -> Command<Self::Message> {
        match msg {}

        //Command::none()
    }

    fn subscription(&self) -> Subscription<Self::Message> { Subscription::none() }

    fn view(&mut self) -> Element<Self::Message> {
        Scrollable::new(&mut self.scroll)
            .align_items(Align::Center)
            .width(Length::Fill)
            .height(Length::Fill)
            .push(
                Image::new(self.map_image.clone())
                    .width(Length::Units(4096))
                    .height(Length::Units(4096)),
            )
            .into()
    }
}

fn server_worker(server: Arc<Mutex<Server>>) {
    const TPS: u64 = 30;
    const RUST_LOG_ENV: &str = "RUST_LOG";

    info!("Starting server...");

    // Set up an fps clock
    let mut clock = Clock::start();

    info!("Server is ready to accept connections.");
    let game_addr = server.lock().unwrap().settings().gameserver_address.port();
    info!(?game_addr, "starting server at port");
    let metrics_addr = server.lock().unwrap().settings().metrics_address.port();
    info!(?metrics_addr, "starting metrics at port");

    loop {
        let events = {
            let mut server = server.lock().unwrap();

            let events = server
                .tick(Input::default(), clock.get_last_delta())
                .expect("Failed to tick server");

            // Clean up the server after a tick.
            server.cleanup();

            events
        };

        for event in events {
            match event {
                Event::ClientConnected { entity: _ } => info!("Client connected!"),
                Event::ClientDisconnected { entity: _ } => info!("Client disconnected!"),
                Event::Chat { entity: _, msg } => info!("[Client] {}", msg),
            }
        }

        // Wait for the next tick.
        clock.tick(Duration::from_millis(1000 / TPS));
    }
}

const TPS: u64 = 30;
const RUST_LOG_ENV: &str = "RUST_LOG";

fn main() {
    // Init logging
    let filter = match std::env::var_os(RUST_LOG_ENV).map(|s| s.into_string()) {
        Some(Ok(env)) => {
            let mut filter = EnvFilter::new("veloren_world::sim=info")
                .add_directive("veloren_world::civ=info".parse().unwrap())
                .add_directive(LevelFilter::INFO.into());
            for s in env.split(',').into_iter() {
                match s.parse() {
                    Ok(d) => filter = filter.add_directive(d),
                    Err(err) => println!("WARN ignoring log directive: `{}`: {}", s, err),
                };
            }
            filter
        },
        _ => EnvFilter::from_env(RUST_LOG_ENV)
            .add_directive("veloren_world::sim=info".parse().unwrap())
            .add_directive("veloren_world::civ=info".parse().unwrap())
            .add_directive(LevelFilter::INFO.into()),
    };

    FmtSubscriber::builder()
        .with_max_level(Level::ERROR)
        .with_env_filter(filter)
        .init();

    Gui::run(Settings::default())
}
