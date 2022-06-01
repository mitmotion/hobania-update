use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use vek::*;

// create pseudorandom from position
fn seed_from_pos(pos: Vec3<i32>) -> [u8; 32] {
    [
        pos.x as u8,
        (pos.x >> 8) as u8,
        (pos.x >> 16) as u8,
        (pos.x >> 24) as u8,
        0,
        0,
        0,
        0,
        pos.y as u8,
        (pos.y >> 8) as u8,
        (pos.y >> 16) as u8,
        (pos.y >> 24) as u8,
        0,
        0,
        0,
        0,
        pos.z as u8,
        (pos.z >> 8) as u8,
        (pos.z >> 16) as u8,
        (pos.z >> 24) as u8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
}

struct FireplaceTiming {
    // all this assumes sunrise at 6am, sunset at 6pm
    breakfast: f32,   // 5am to 7am
    dinner: f32,      // 5pm to 7pm
    daily_cycle: f32, // 30min to 2hours
}

const SMOKE_BREAKFAST_STRENGTH: f32 = 96.0;
const SMOKE_BREAKFAST_HALF_DURATION: f32 = 45.0 * 60.0;
const SMOKE_BREAKFAST_START: f32 = 5.0 * 60.0 * 60.0;
const SMOKE_BREAKFAST_RANGE: f32 = 2.0 * 60.0 * 60.0;
const SMOKE_DINNER_STRENGTH: f32 = 128.0;
const SMOKE_DINNER_HALF_DURATION: f32 = 60.0 * 60.0;
const SMOKE_DINNER_START: f32 = 17.0 * 60.0 * 60.0;
const SMOKE_DINNER_RANGE: f32 = 2.0 * 60.0 * 60.0;
const SMOKE_DAILY_CYCLE_MIN: f32 = 30.0 * 60.0;
const SMOKE_DAILY_CYCLE_MAX: f32 = 120.0 * 60.0;
const SMOKE_MAX_TEMPERATURE: f32 = 0.0; // temperature for nominal smoke (0..daily_var)
const SMOKE_MAX_TEMP_VALUE: f32 = 1.0;
const SMOKE_TEMP_MULTIPLIER: f32 = 96.0;
const SMOKE_DAILY_VARIATION: f32 = 32.0;

struct FireplaceClimate {
    daily_strength: f32, // can be negative (offset)
    day_start: f32,      // seconds since breakfast for daily cycle
    day_end: f32,        // seconds before dinner on daily cycle
}

fn create_timing(rng: &mut ChaCha8Rng) -> FireplaceTiming {
    let breakfast: f32 = SMOKE_BREAKFAST_START + rng.gen::<f32>() * SMOKE_BREAKFAST_RANGE;
    let dinner: f32 = SMOKE_DINNER_START + rng.gen::<f32>() * SMOKE_DINNER_RANGE;
    let daily_cycle: f32 =
        SMOKE_DAILY_CYCLE_MIN + rng.gen::<f32>() * (SMOKE_DAILY_CYCLE_MAX - SMOKE_DAILY_CYCLE_MIN);
    FireplaceTiming {
        breakfast,
        dinner,
        daily_cycle,
    }
}

fn create_climate(temperature: f32, _humidity: f32) -> FireplaceClimate {
    // temp -1…1, humidity 0…1
    let daily_strength =
        (SMOKE_MAX_TEMPERATURE - temperature).min(SMOKE_MAX_TEMP_VALUE) * SMOKE_TEMP_MULTIPLIER;
    // when is breakfast down to daily strength
    // daily_strength ==
    // SMOKE_BREAKFAST_STRENGTH*(1.0-(t-breakfast)/SMOKE_BREAKFAST_HALF_DURATION)
    // (t-breakfast) = (1.0 -
    // daily_strength/SMOKE_BREAKFAST_STRENGTH)*SMOKE_BREAKFAST_HALF_DURATION
    let day_start = (SMOKE_BREAKFAST_STRENGTH - daily_strength.max(0.0))
        * (SMOKE_BREAKFAST_HALF_DURATION / SMOKE_BREAKFAST_STRENGTH);
    let day_end = (SMOKE_DINNER_STRENGTH - daily_strength.max(0.0))
        * (SMOKE_DINNER_HALF_DURATION / SMOKE_DINNER_STRENGTH);
    FireplaceClimate {
        daily_strength,
        day_start,
        day_end,
    }
}

type Increasing = bool;

pub fn smoke_at_time(
    position: Vec3<i32>,
    temperature: f32,
    humidity: f32,
    time_of_day: f32,
) -> (f32, Increasing) {
    let mut pseudorandom = ChaCha8Rng::from_seed(seed_from_pos(position));
    let timing = create_timing(&mut pseudorandom);
    let climate = create_climate(temperature, humidity);
    let after_breakfast = time_of_day - timing.breakfast;
    //let after_dinner = time_of_day-timing.dinner;
    if after_breakfast < -SMOKE_BREAKFAST_HALF_DURATION {
        /* night */
        (0.0, false)
    } else if after_breakfast < 0.0 {
        /* cooking breakfast */
        (
            (SMOKE_BREAKFAST_HALF_DURATION + after_breakfast)
                * (SMOKE_BREAKFAST_STRENGTH / SMOKE_BREAKFAST_HALF_DURATION),
            true,
        )
    } else if time_of_day < climate.day_start {
        /* cooling */
        (
            (SMOKE_BREAKFAST_HALF_DURATION - after_breakfast)
                * (SMOKE_BREAKFAST_STRENGTH / SMOKE_BREAKFAST_HALF_DURATION),
            false,
        )
    } else if time_of_day < climate.day_end {
        /* day cycle */
        let day_phase = ((time_of_day - climate.day_start) / timing.daily_cycle).fract();
        if day_phase < 0.5 {
            (
                (climate.daily_strength + day_phase * (2.0 * SMOKE_DAILY_VARIATION)).max(0.0),
                true,
            )
        } else {
            (
                (climate.daily_strength + (1.0 - day_phase) * (2.0 * SMOKE_DAILY_VARIATION))
                    .max(0.0),
                false,
            )
        }
    } else if time_of_day < timing.dinner {
        /* cooking dinner */
        (
            (SMOKE_BREAKFAST_HALF_DURATION + time_of_day - timing.dinner)
                * (SMOKE_BREAKFAST_STRENGTH / SMOKE_BREAKFAST_HALF_DURATION),
            true,
        )
    } else {
        /* cooling + night */
        (
            (SMOKE_BREAKFAST_HALF_DURATION - time_of_day + timing.dinner).max(0.0)
                * (SMOKE_BREAKFAST_STRENGTH / SMOKE_BREAKFAST_HALF_DURATION),
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoke() {
        let position = Vec3::new(22_i32, 11, 33);
        let temperature = -0.5;
        let humidity = 0.5;
        for i in (0..24 * 4) {
            let time_of_day = 15.0 * 60.0 * (i as f32);
            println!(
                "{:.2} {}",
                time_of_day / (60.0 * 60.0),
                smoke_at_time(position, temperature, humidity, time_of_day)
            );
        }
    }
}
