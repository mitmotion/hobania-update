use common::{
    grid::Grid,
    resources::TimeOfDay,
    weather::{Weather, WeatherGrid, CELL_SIZE, CHUNKS_PER_CELL},
};
use noise::{NoiseFn, SuperSimplex, Turbulence};
use vek::*;
use world::World;

use crate::weather::WEATHER_DT;

fn cell_to_wpos_center(p: Vec2<i32>) -> Vec2<i32> { p * CELL_SIZE as i32 + CELL_SIZE as i32 / 2 }

#[derive(Clone)]
struct WeatherZone {
    weather: Weather,
    /// Time, in seconds this zone lives.
    time_to_live: f32,
}

struct CellConsts {
    rain_factor: f32,
}

pub struct WeatherSim {
    size: Vec2<u32>,
    consts: Grid<CellConsts>,
    zones: Grid<Option<WeatherZone>>,
}

impl WeatherSim {
    pub fn new(size: Vec2<u32>, world: &World) -> Self {
        Self {
            size,
            consts: Grid::from_raw(
                size.as_(),
                (0..size.x * size.y)
                    .map(|i| Vec2::new(i % size.x, i / size.x))
                    .map(|p| {
                        let mut humid_sum = 0.0;

                        for y in 0..CHUNKS_PER_CELL {
                            for x in 0..CHUNKS_PER_CELL {
                                let chunk_pos = p * CHUNKS_PER_CELL + Vec2::new(x, y);
                                if let Some(chunk) = world.sim().get(chunk_pos.as_()) {
                                    let env = chunk.get_environment();
                                    humid_sum += env.humid;
                                }
                            }
                        }
                        let average_humid = humid_sum / (CHUNKS_PER_CELL * CHUNKS_PER_CELL) as f32;
                        let rain_factor = (4.0 * average_humid.powf(0.2)).min(1.0);
                        CellConsts { rain_factor }
                    })
                    .collect::<Vec<_>>(),
            ),
            zones: Grid::new(size.as_(), None),
        }
    }

    /// Adds a weather zone as a circle at a position, with a given radius. Both
    /// of which should be in weather cell units
    pub fn add_zone(&mut self, weather: Weather, pos: Vec2<f32>, radius: f32, time: f32) {
        let min: Vec2<i32> = (pos - radius).as_::<i32>().map(|e| e.max(0));
        let max: Vec2<i32> = (pos + radius)
            .ceil()
            .as_::<i32>()
            .map2(self.size.as_::<i32>(), |a, b| a.min(b));
        for y in min.y..max.y {
            for x in min.x..max.x {
                let ipos = Vec2::new(x, y);
                let p = ipos.as_::<f32>();

                if p.distance_squared(pos) < radius.powi(2) {
                    self.zones[ipos] = Some(WeatherZone {
                        weather,
                        time_to_live: time,
                    });
                }
            }
        }
    }

    // Time step is cell size / maximum wind speed
    pub fn tick(&mut self, world: &World, time_of_day: &TimeOfDay, out: &mut WeatherGrid) {
        let time = time_of_day.0;

        let base_nz = Turbulence::new(
            Turbulence::new(SuperSimplex::new())
                .set_frequency(0.2)
                .set_power(1.5),
        )
        .set_frequency(2.0)
        .set_power(0.2);

        let rain_nz = SuperSimplex::new();

        for (point, cell) in out.iter_mut() {
            if let Some(zone) = &mut self.zones[point] {
                *cell = zone.weather;
                zone.time_to_live -= WEATHER_DT;
                if zone.time_to_live <= 0.0 {
                    self.zones[point] = None;
                }
            } else {
                let wpos = cell_to_wpos_center(point);

                let humid = world
                    .sim()
                    .get_wpos(wpos)
                    .map_or(1.0, |c| c.get_environment().humid);

                let pos = wpos.as_::<f64>() + time as f64 * 0.1;

                let space_scale = 7_500.0;
                let time_scale = 100_000.0;
                let spos = (pos / space_scale).with_z(time as f64 / time_scale);

                let pressure =
                    (base_nz.get(spos.into_array()) * 0.5 + 1.0).clamped(0.0, 1.0) as f32 + 0.4
                        - humid * 0.5;

                const RAIN_CLOUD_THRESHOLD: f32 = 0.2;
                cell.cloud = (1.0 - pressure).max(0.0) * 0.5;
                cell.rain = (1.0 - pressure - RAIN_CLOUD_THRESHOLD).max(0.0)
                    * self.consts[point].rain_factor;
                cell.wind = Vec2::new(
                    rain_nz.get(spos.into_array()).powi(3) as f32,
                    rain_nz.get((spos + 1.0).into_array()).powi(3) as f32,
                ) * 200.0
                    * (1.0 - pressure);
            }
        }
    }

    pub fn size(&self) -> Vec2<u32> { self.size }
}
