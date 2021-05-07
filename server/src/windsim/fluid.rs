use super::types::*;
/// Macro for indexing into a 1D array using 3D coordinates.
macro_rules! IX {
    ( $x: expr, $y: expr,  $z: expr ) => {{ $x as usize + (X_SIZE + 2) * ($y as usize + (Y_SIZE + 2) * $z as usize) }};
}

/// 
/// * `b` - The type of border. 1 for vertical vel walls, 2 for side vel walls,
///   3 for back-front walls, 0 for dens.
fn set_borders(grid: &mut Box<[f32]>, b: u8) {
    let max_x = X_SIZE + 1;
    let max_y = Y_SIZE + 1;
    let max_z = Z_SIZE + 1;

    for ii in 0..max_x {
        for kk in 0..max_z {
            let ix_top = IX!(ii, max_y, kk);
            let ix_top_inset = IX!(ii, max_y - 1, kk);
            let ix_bot = IX!(ii, 0, kk);
            let ix_bot_inset = IX!(ii, 1, kk);
            grid[ix_top] = {
                if b == 2 {
                    -grid[ix_top_inset]
                } else {
                    grid[ix_top_inset]
                }
            };
            grid[ix_bot] = {
                if b == 2 {
                    -grid[ix_bot_inset]
                } else {
                    grid[ix_bot_inset]
                }
            };
        }
    }
    // SIDE WALLS
    for jj in 0..max_y {
        for kk in 0..max_z {
            let ix_left = IX!(0, jj, kk);
            let ix_left_inset = IX!(1, jj, kk);
            let ix_right = IX!(max_x, jj, kk);
            let ix_right_inset = IX!(max_x - 1, jj, kk);
            grid[ix_left] = {
                if b == 1 {
                    -grid[ix_left_inset]
                } else {
                    grid[ix_left_inset]
                }
            };
            grid[ix_right] = {
                if b == 1 {
                    -grid[ix_right_inset]
                } else {
                    grid[ix_right_inset]
                }
            };
        }
    }
    // BACK - FRONT WALLS
    for ii in 0..max_x {
        for jj in 0..max_y {
            let ix_front = IX!(ii, jj, 0);
            let ix_front_inset = IX!(ii, jj, 1);
            let ix_back = IX!(ii, jj, max_z);
            let ix_back_inset = IX!(ii, jj, max_z - 1);
            grid[ix_front] = {
                if b == 3 {
                    -grid[ix_front_inset]
                } else {
                    grid[ix_front_inset]
                }
            };
            grid[ix_back] = {
                if b == 3 {
                    -grid[ix_back_inset]
                } else {
                    grid[ix_back_inset]
                }
            };
        }
    }

    // For the 12 edges of the 3d grid
    for ii in 1..max_x - 1 {
        grid[IX!(ii, 0, 0)] = (grid[IX!(ii + 1, 0, 0)]
            + grid[IX!(ii - 1, 0, 0)]
            + grid[IX!(ii, 1, 0)]
            + grid[IX!(ii, 0, 1)])
            / 4.0;
        grid[IX!(ii, max_y, 0)] = (grid[IX!(ii + 1, max_y, 0)]
            + grid[IX!(ii - 1, max_y, 0)]
            + grid[IX!(ii, max_y - 1, 0)]
            + grid[IX!(ii, max_y, 1)])
            / 4.0;
        grid[IX!(ii, 0, max_z)] = (grid[IX!(ii + 1, 0, max_z)]
            + grid[IX!(ii - 1, 0, max_z)]
            + grid[IX!(ii, 0, max_z - 1)]
            + grid[IX!(ii, 1, max_z)])
            / 4.0;
        grid[IX!(ii, max_y, max_z)] = (grid[IX!(ii + 1, max_y, max_z)]
            + grid[IX!(ii - 1, max_y, max_z)]
            + grid[IX!(ii, max_y - 1, max_z)]
            + grid[IX!(ii, max_y, max_z - 1)])
            / 4.0;
    }
    // The 8 vertices of the 3d grid
    //grid[IX!(0, 0, 0)] = (grid[IX!(1, 0, Z_SIZE - 1)] + grid[IX!(0, 1, Z_SIZE
    // - 1)]);
}

fn linear_solver(
    grid: &mut Box<[f32]>,
    prev_grid: &Box<[f32]>,
    diff_rate: f32,
    denominator: f32,
    borders: bool,
    b: u8,
) {
    // For each cell we get contributions from all 6 direct neighbors
    let iterations = 1;

    for _ in 0..iterations {
        for ii in 1..=X_SIZE {
            for jj in 1..=Y_SIZE {
                for kk in 1..=Z_SIZE {
                    let ix = IX!(ii, jj, kk); //Index of current cell
                    let ix_up = IX!(ii, jj - 1, kk); // 1 row up
                    let ix_down = IX!(ii, jj + 1, kk); // 1 row down
                    let ix_back = IX!(ii, jj, kk - 1); // 1 row back
                    let ix_front = IX!(ii, jj, kk + 1); // 1 row front
                    grid[ix] = (prev_grid[ix]
                        + diff_rate
                            * (grid[ix - 1]
                                + grid[ix + 1]
                                + grid[ix_up]
                                + grid[ix_down]
                                + grid[ix_back]
                                + grid[ix_front]))
                        / denominator;
                }
            }
        }
        if borders {
            set_borders(grid, b)
        }
    }
}

fn trace_backwards(pos: f32, distance_moved: f32, size: f32) -> (f32, f32, f32, f32) {
    //estimated position in previous step (not on a cell center - cell centers are
    // integers)
    let mut prev_pos = pos - distance_moved;

    // is the estimated position past the edges? keep inside
    if prev_pos < 0.5 {
        prev_pos = 0.5
    }
    if prev_pos > size + 0.5 {
        prev_pos = size + 0.5
    }

    // find cell centers to left and right (or up and down, etc)
    let p0 = prev_pos.floor();
    let p1 = prev_pos.ceil();

    // how close to cell centers? 0 to 1
    let d1 = prev_pos - p0;
    let d0 = 1.0 - d1;

    // return adjacent cell centers and distances to centers
    (p0, p1, d0, d1)
}

/// Diffuse values with neighbors. Depends on viscosity (non-viscosity).
fn diffuse(
    grid: &mut Box<[f32]>,
    prev_grid: &Box<[f32]>,
    dt: f32,
    viscosity: f32,
    borders: bool,
    b: u8,
) {
    let diff_rate = dt * viscosity * grid.len() as f32;
    if viscosity == 0.0 {
        return;
    }
    linear_solver(
        grid,
        prev_grid,
        diff_rate,
        1.0 + 6.0 * diff_rate,
        borders,
        b,
    )
}

/// Move values accross velocity fields
#[allow(clippy::too_many_arguments)]
fn advect(
    grid: &mut Box<[f32]>,
    prev_grid: &Box<[f32]>,
    vx_grid: &Box<[f32]>,
    vy_grid: &Box<[f32]>,
    vz_grid: &Box<[f32]>,
    dt: f32,
    borders: bool,
    b: u8,
) {
    let dt0x = dt * X_SIZE as f32;
    let dt0y = dt * Y_SIZE as f32;
    let dt0z = dt * Z_SIZE as f32;

    for ii in 1..=X_SIZE {
        for jj in 1..=Y_SIZE {
            for kk in 1..=Z_SIZE {
                // current cell center
                let ix = IX!(ii, jj, kk);

                // positions of and distance to adjacent cell centers to previous position of
                // current cell center
                let (x0, x1, r0, r1) =
                    trace_backwards(ii as f32, dt0x * vx_grid[ix], X_SIZE as f32);
                let (y0, y1, s0, s1) =
                    trace_backwards(jj as f32, dt0y * vy_grid[ix], Y_SIZE as f32);
                let (z0, z1, t0, t1) =
                    trace_backwards(kk as f32, dt0z * vz_grid[ix], Z_SIZE as f32);

                // all adjacent cell centers surrounding previous position (like vertices of a
                // cube)
                let ix000 = IX!(x0, y0, z0);
                let ix010 = IX!(x0, y1, z0);
                let ix100 = IX!(x1, y0, z0);
                let ix110 = IX!(x1, y1, z0);
                let ix001 = IX!(x0, y0, z1);
                let ix011 = IX!(x0, y1, z1);
                let ix111 = IX!(x1, y1, z1);
                let ix101 = IX!(x1, y0, z1);

                // value of cell is weighed average of the values of the 8 cell centers
                grid[ix] = r0
                    * (s0 * (t0 * prev_grid[ix000] + t1 * prev_grid[ix001])
                        + s1 * (t0 * prev_grid[ix010] + t1 * prev_grid[ix011]))
                    + r1 * (s0 * (t0 * prev_grid[ix100] + t1 * prev_grid[ix101])
                        + s1 * (t0 * prev_grid[ix110] + t1 * prev_grid[ix111]));
            }
        }
    }
    if borders {
        set_borders(grid, b);
    }
}

// Forces velocity to be mass conserving
#[allow(clippy::too_many_arguments)]
fn project(
    vx_grid: &mut Box<[f32]>,
    vy_grid: &mut Box<[f32]>,
    vz_grid: &mut Box<[f32]>,
    prev_x: &mut Box<[f32]>,
    prev_y: &mut Box<[f32]>,
    prev_z: &mut Box<[f32]>,
    borders: bool,
) {
    for ii in 1..=X_SIZE {
        for jj in 1..=Y_SIZE {
            for kk in 1..=Z_SIZE {
                let ix = IX!(ii, jj, kk);
                let ix_up = IX!(ii, jj - 1, kk); // 1 row up
                let ix_down = IX!(ii, jj + 1, kk); // 1 row down
                let ix_back = IX!(ii, jj, kk - 1); // 1 row back
                let ix_front = IX!(ii, jj, kk + 1); // 1 row front
                prev_y[ix] = -0.5
                    * ((vx_grid[ix + 1] - vx_grid[ix - 1]) / X_SIZE as f32
                        + (vy_grid[ix_down] - vy_grid[ix_up]) / Y_SIZE as f32
                        + (vz_grid[ix_front] - vz_grid[ix_back]) / Z_SIZE as f32);
                prev_x[ix] = 0.0;
            }
        }
    }
    if borders {
        set_borders(prev_y, 1);
        set_borders(prev_x, 2);
        set_borders(prev_z, 3);
    }

    //Gauss seidel to compute gradient field x with y (ignoring z right now)
    linear_solver(prev_x, prev_y, 1.0, 6.0, false, 0);

    //Substract gradient field
    for ii in 1..=X_SIZE {
        for jj in 1..=Y_SIZE {
            for kk in 1..=Z_SIZE {
                let ix = IX!(ii, jj, kk);
                let ix_up = IX!(ii, jj - 1, kk); // 1 row up
                let ix_down = IX!(ii, jj + 1, kk); // 1 row down
                let ix_back = IX!(ii, jj, kk - 1); // 1 row back
                let ix_front = IX!(ii, jj, kk + 1); // 1 row front
                vx_grid[ix] -= 0.5 * (prev_x[ix + 1] - prev_x[ix - 1]) * X_SIZE as f32;
                vy_grid[ix] -= 0.5 * (prev_x[ix_down] - prev_x[ix_up]) * Y_SIZE as f32;
                vz_grid[ix] -= 0.5 * (prev_x[ix_front] - prev_x[ix_back]) * Z_SIZE as f32;
            }
        }
    }
    if borders {
        set_borders(vx_grid, 1);
        set_borders(vy_grid, 2);
        set_borders(vz_grid, 3);
    }
}

/// Step density
fn step_dens(
    dens_grid: &mut Box<[f32]>,
    vx_grid: &mut Box<[f32]>,
    vy_grid: &mut Box<[f32]>,
    vz_grid: &mut Box<[f32]>,
    dt: f32,
    diff: f32,
    borders: bool,
) {
    // Make a copy of the dens_grid
    let prev_dens_grid = &mut dens_grid.clone();

    // Swap
    let (prev_dens_grid, dens_grid) = (dens_grid, prev_dens_grid);

    // Diffuse
    diffuse(dens_grid, prev_dens_grid, dt, diff, borders, 0);

    // Swap
    let (prev_dens_grid, dens_grid) = (dens_grid, prev_dens_grid);

    // Advect
    advect(
        dens_grid,
        prev_dens_grid,
        vx_grid,
        vy_grid,
        vz_grid,
        dt,
        borders,
        0,
    );
}

/// Step velocity
fn step_vel(
    vx_grid: &mut Box<[f32]>,
    vy_grid: &mut Box<[f32]>,
    vz_grid: &mut Box<[f32]>,
    dt: f32,
    diff: f32,
    borders: bool,
) {
    let prev_x = &mut vx_grid.clone();
    let prev_y = &mut vy_grid.clone();
    let prev_z = &mut vz_grid.clone();

    // Swap grids
    let (prev_x, vx_grid) = (vx_grid, prev_x);
    let (prev_y, vy_grid) = (vy_grid, prev_y);
    let (prev_z, vz_grid) = (vz_grid, prev_z);

    // Diffuse just like with density but with velocity instead
    diffuse(vx_grid, prev_x, dt, diff, borders, 1);
    diffuse(vy_grid, prev_y, dt, diff, borders, 2);
    diffuse(vz_grid, prev_z, dt, diff, borders, 3);

    // For mass conservation before advect
    project(vx_grid, vy_grid, vz_grid, prev_x, prev_y, prev_z, borders);

    // Swap grids
    let (prev_x, vx_grid) = (vx_grid, prev_x);
    let (prev_y, vy_grid) = (vy_grid, prev_y);
    let (prev_z, vz_grid) = (vz_grid, prev_z);

    // Advect just like with density
    advect(vx_grid, prev_x, prev_x, prev_y, prev_z, dt, borders, 1);
    advect(vy_grid, prev_y, prev_x, prev_y, prev_z, dt, borders, 2);
    advect(vz_grid, prev_z, prev_x, prev_y, prev_z, dt, borders, 2);

    project(vx_grid, vy_grid, vz_grid, prev_x, prev_y, prev_z, borders);
}

pub fn step_fluid(
    dens_grid: &mut Box<[f32]>,
    vx_grid: &mut Box<[f32]>,
    vy_grid: &mut Box<[f32]>,
    vz_grid: &mut Box<[f32]>,
    dt: f32,
    viscosity: f32,
    borders: bool,
) {
    step_dens(dens_grid, vx_grid, vy_grid, vz_grid, dt, viscosity, borders);
    step_vel(vx_grid, vy_grid, vz_grid, dt, viscosity, borders);
}
