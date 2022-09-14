use common::comp;
use hashbrown::HashSet;
use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    thread,
    time::{Duration, SystemTime},
};
use structopt::StructOpt;
use tokio::runtime::{self, Runtime};
use vek::*;
use common_state::State;
use veloren_client::{addr::ConnectionArgs, Client};

#[derive(Clone, Copy, StructOpt)]
struct Opt {
    /// Number of clients to spin up
    size: u32,
    /// View distance of each client
    vd: u32,
    /// Distribution of the clients, if not clustered they are dispersed
    #[structopt(short, long)]
    clustered: bool,
    /// Whether the clients should move
    #[structopt(short, long)]
    movement: bool,
}

fn main() {
    let opt = Opt::from_args();
    // Start logging
    let _guards = common_frontend::init_stdout(None);
    // Run clients and stuff
    //
    // NOTE:  "swarm0" is assumed to be an admin already
    //
    // Since this requires a no-auth server use this command to add swarm0 as an
    // admin:
    //
    // --no-auth admin add swarm0 Admin
    //
    let admin_username = "swarm0".to_owned();
    let usernames = (1..opt.size)
        .map(|i| format!("swarm{}", i))
        .collect::<Vec<_>>();
    let to_adminify = usernames.clone();

    let finished_init = Arc::new(AtomicU32::new(0));
    let mut builder = runtime::Builder::new_multi_thread();
    builder
        .max_blocking_threads(opt.size as usize + 1)
        .thread_name("swarm");
    let mut pools = common_state::State::pools(common::resources::GameMode::Client, builder);
    pools.slowjob_threads = 0;

    // TODO: calculate and log the required chunks per second to maintain the
    // selected scenario with full vd loaded

    run_client_new_thread(
        admin_username,
        0,
        to_adminify,
        &pools,
        opt,
        &finished_init,
    );

    usernames.into_iter().enumerate().for_each(|(index, name)| {
        run_client_new_thread(
            name,
            index as u32,
            Vec::new(),
            &pools,
            opt,
            &finished_init,
        );
    });

    std::thread::park();
}

fn run_client_new_thread(
    username: String,
    index: u32,
    to_adminify: Vec<String>,
    pools: &common_state::Pools,
    opt: Opt,
    finished_init: &Arc<AtomicU32>,
) {
    let pools_ = pools.clone();
    let finished_init = Arc::clone(finished_init);
    pools.runtime.spawn(async move {
        if let Err(err) = run_client(username, index, to_adminify, pools_, opt, finished_init).await {
            tracing::error!("swarm member {} exited with an error: {:?}", index, err);
        }
    });
}

async fn run_client(
    username: String,
    index: u32,
    to_adminify: Vec<String>,
    pools: common_state::Pools,
    opt: Opt,
    finished_init: Arc<AtomicU32>,
) -> Result<(), veloren_client::Error> {
    let mut clock = common::clock::Clock::new(Duration::from_secs_f32(1.0 / 30.0));
    async fn tick<'a>(client: &'a mut Client, clock: &'a mut common::clock::Clock) -> Result<(), veloren_client::Error> {
        clock.tick_slow().await;
        client.tick_network(clock.dt())?;
        client.cleanup();
        Ok(())
    }

    let mut client = loop {
        let pools = &pools;
        let to_adminify = &*to_adminify;
        let username = &username;
        let clock = &mut clock;
        let run = move || async move {
            // Connect to localhost
            let addr = ConnectionArgs::Tcp {
                prefer_ipv6: false,
                hostname: "localhost".into(),
            };
            // NOTE: use a no-auth server
            let mut client = Client::new(
                    addr,
                    &mut None,
                    pools.clone(),
                    &username,
                    "",
                    |_| false,
                ).await?;
            tracing::info!("Client {} connected", index);

            // Wait for character list to load
            client.load_character_list();
            while client.character_list().loading {
                tick(&mut client, clock).await?;
            }
            tracing::info!("Client {} loaded character list", index);

            // Create character if none exist
            if client.character_list().characters.is_empty() {
                client.create_character(
                    username.clone(),
                    Some("common.items.weapons.sword.starter".into()),
                    None,
                    body(),
                );

                client.load_character_list();

                while client.character_list().loading || client.character_list().characters.is_empty() {
                    tick(&mut client, clock).await?;
                }
            }
            tracing::info!("Client {} found or created character", index);

            client.set_view_distance(opt.vd);

            // Select the first character
            client.request_character(
                client
                    .character_list()
                    .characters
                    .first()
                    .expect("Just created new character if non were listed!!!")
                    .character
                    .id
                    .expect("Why is this an option?"),
            );

            // If this is the admin client then adminify the other swarm members
            if !to_adminify.is_empty() {
                // Wait for other clients to connect
                loop {
                    tick(&mut client, clock).await?;
                    // NOTE: it's expected that each swarm member will have a unique alias
                    let players = client.players().collect::<HashSet<&str>>();
                    if to_adminify
                        .iter()
                        .all(|name| players.contains(&name.as_str()))
                    {
                        break;
                    }
                }
                // Assert that we are a moderator (assumes we are an admin if so)
                assert!(
                    client.is_moderator(),
                    "The user needs to ensure \"{}\" is registered as an admin on the server",
                    username
                );
                // Send commands to adminify others
                to_adminify.iter().for_each(|name| {
                    client.send_command("adminify".into(), vec![name.into(), "admin".into()])
                });
            }

            // Wait for moderator
            while !client.is_moderator() {
                tick(&mut client, clock).await?;
            }
            client.clear_terrain();
            client.request_player_physics(false);

            Ok::<_, veloren_client::Error>(client)
        };

        match run().await {
            Err(e) => {
                tokio::time::sleep(Duration::from_secs(1));
                tracing::warn!(?e, "Client {} disconnected", index)
            },
            Ok(client) => {
                break client
            },
        }
    };
    drop(pools);

    finished_init.fetch_add(1, Ordering::Relaxed);
    // Wait for initialization of all other swarm clients to finish
    while finished_init.load(Ordering::Relaxed) != opt.size {
        tick(&mut client, &mut clock).await?;
    }

    // Use this check so this is only printed once
    if !to_adminify.is_empty() {
        println!("Initialization of all clients finished!");
    }

    // Main loop
    let chunk_size = 32.0; // TODO: replace with the actual constant
    let world_center = client
        .world_data()
        .chunk_size()
        .map(|e| e as f32 * chunk_size)
        / 2.0;
    loop {
        // TODO: doesn't seem to produce an error when server is shutdown (process keeps
        // running)
        tick(&mut client, &mut clock).await?;
        let entity = client.entity();
        // Move or stay still depending on specified options
        // TODO: make sure server cheat protections aren't triggering
        let pos = comp::Pos(position(index, opt) + world_center);
        let vel = comp::Vel(Default::default());
        client
            .state_mut()
            .write_component_ignore_entity_dead(entity, pos);
        client
            .state_mut()
            .write_component_ignore_entity_dead(entity, vel);
    }
}

// Use client index, opts, and current system time to determine position
fn position(index: u32, opt: Opt) -> Vec3<f32> {
    // TODO: replace 32 with constant for chunk size
    let chunk_size = 32.0;

    let width = (opt.size as f32).sqrt().round() as u32;

    let spacing = if opt.clustered {
        5.0
    } else {
        use common::region::REGION_SIZE;
        // Attempt to make regions subscribed to by each client not overlapping
        opt.vd as f32 * 2.0 * chunk_size + 2.0 * REGION_SIZE as f32
    };

    // Offset to center the grid of clients
    let offset = Vec2::new(
        width as f32 * spacing / 2.0,
        (opt.size / width) as f32 / 2.0,
    );
    // Position clients in a grid
    let base_pos = Vec2::new(
        (index % width) as f32 * spacing,
        (index / width) as f32 * spacing,
    ) - offset;

    let movement_offset: Vec2<_> = if opt.movement {
        // blocks per second
        const SPEED: f32 = 9.0; // typical super fast veloren walking speed

        // move in a square route
        // in blocks
        let route_side_length = chunk_size * opt.vd as f32 * 3.0;
        let route_length = route_side_length * 4.0;
        // in secs
        let route_time = route_length / SPEED;
        let route_progress = (SystemTime::UNIX_EPOCH.elapsed().unwrap().as_secs_f64()
            % route_time as f64) as f32
            / route_time;

        // clockwise square
        (match route_progress * 4.0 {
            // going up left side
            t if t < 1.0 => Vec2::new(0.0, 0.0 + t),
            // going across top side
            t if t < 2.0 => Vec2::new(0.0 + (t - 1.0), 1.0),
            // going down right side
            t if t < 3.0 => Vec2::new(1.0, 1.0 - (t - 2.0)),
            // going across bottom
            t => Vec2::new(1.0 - (t - 3.0), 0.0),
        }) * route_side_length
    } else {
        Vec2::zero()
    };

    Vec3::from(base_pos + movement_offset)
}

fn body() -> comp::Body {
    comp::body::humanoid::Body {
        species: comp::body::humanoid::Species::Human,
        body_type: comp::body::humanoid::BodyType::Male,
        hair_style: 0,
        beard: 0,
        eyes: 0,
        accessory: 0,
        hair_color: 0,
        skin: 0,
        eye_color: 0,
    }
    .into()
}
