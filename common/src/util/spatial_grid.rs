use core::sync::atomic::{AtomicU32, Ordering};
use vek::*;

pub type MapMut = dashmap::DashMap<Vec2<i32>, Vec<specs::Entity>>;
pub type MapRef = dashmap::ReadOnlyView<Vec2<i32>, Vec<specs::Entity>>;

#[derive(Debug)]
pub struct SpatialGridInner<Map, Radius> {
    /// Uses two scales of grids so that we can have a hard limit on how far to search in the
    /// smaller grid
    grid: Map,
    large_grid: Map,
    /// Log base 2 of the cell size of the spatial grid
    lg2_cell_size: usize,
    /// Log base 2 of the cell size of the large spatial grid
    lg2_large_cell_size: usize,
    /// Entities with a radius over this value are store in the coarser large_grid
    /// This is the amount of buffer space we need to add when finding the intersections with cells
    /// in the regular grid
    radius_cutoff: u32,
    /// Stores the largest radius of the entities in the large_grid
    /// This is the amount of buffer space we need to add when finding the intersections with cells
    /// in the larger grid
    /// note: could explore some distance field type thing for querying whether there are large
    /// entities nearby that necessitate expanding the cells searched for collision (and querying
    /// how much it needs to be expanded)
    /// TODO: log this to metrics?
    largest_large_radius: Radius,
}

pub type SpatialGrid = SpatialGridInner<MapMut, AtomicU32>;
pub type SpatialGridRef = SpatialGridInner<MapRef, u32>;

impl SpatialGrid {
    pub fn new(lg2_cell_size: usize, lg2_large_cell_size: usize, radius_cutoff: u32) -> Self {
        Self {
            grid: Default::default(),
            large_grid: Default::default(),
            lg2_cell_size,
            lg2_large_cell_size,
            radius_cutoff,
            largest_large_radius: radius_cutoff.into(),
        }
    }

    /// Add an entity at the provided 2d pos into the spatial grid
    pub fn insert(&self, pos: Vec2<i32>, radius: u32, entity: specs::Entity) {
        if radius <= self.radius_cutoff {
            let cell = pos.map(|e| e >> self.lg2_cell_size);
            self.grid.entry(cell).or_default().push(entity);
        } else {
            let cell = pos.map(|e| e >> self.lg2_large_cell_size);
            self.large_grid.entry(cell).or_default().push(entity);
            // NOTE: Relaxed ordering is sufficient, because max is a monotonic function (for the
            // duration of shared write access to the map; clear takes &mut so it's okay that it's
            // not monotonic).  We don't need to order this operation with anything else, either,
            // because all interesting map operations require unique access to Self to perform,
            // which axiomatically synchronizes with the end of all prior shared borrows.
            //
            // TODO: Verify that intrinsics lower intelligently to a priority update on CPUs (since
            // the intrinsic seems targeted at GPUs).
            self.largest_large_radius.fetch_max(radius, Ordering::Relaxed);
        }
    }

    pub fn into_read_only(self) -> SpatialGridRef {
        SpatialGridInner {
            grid: self.grid.into_read_only(),
            large_grid: self.large_grid.into_read_only(),
            lg2_cell_size: self.lg2_cell_size,
            lg2_large_cell_size: self.lg2_large_cell_size,
            radius_cutoff: self.radius_cutoff,
            largest_large_radius: self.largest_large_radius.into_inner(),
        }
    }

    pub fn clear(&mut self) {
        self.grid.clear();
        self.large_grid.clear();
        self.largest_large_radius = self.radius_cutoff.into();
    }
}

impl SpatialGridRef {
    /// Get an iterator over the entities overlapping the provided axis aligned
    /// bounding region.
    /// NOTE: for best optimization of the iterator use
    /// `for_each` rather than a for loop.
    pub fn in_aabr<'a>(&'a self, aabr: Aabr<i32>) -> impl Iterator<Item = specs::Entity> + 'a {
        let iter = |max_entity_radius, grid: &'a MapRef, lg2_cell_size| {
            // Add buffer for other entity radius
            let min = aabr.min - max_entity_radius as i32;
            let max = aabr.max + max_entity_radius as i32;
            // Convert to cells
            let min = min.map(|e| e >> lg2_cell_size);
            let max = max.map(|e| (e + (1 << lg2_cell_size) - 1) >> lg2_cell_size);

            (min.x..=max.x)
                .flat_map(move |x| (min.y..=max.y).map(move |y| Vec2::new(x, y)))
                .flat_map(move |cell| grid.get(&cell).into_iter().flatten())
                .copied()
        };

        iter(self.radius_cutoff, &self.grid, self.lg2_cell_size).chain(iter(
            self.largest_large_radius,
            &self.large_grid,
            self.lg2_large_cell_size,
        ))
    }

    /// Get an iterator over the entities overlapping the
    /// axis aligned bounding region that contains the provided circle
    /// NOTE: for best optimization of the iterator use `for_each` rather than a
    /// for loop
    // TODO: using the circle directly would be tighter (how efficient would it be
    // to query the cells intersecting a circle?) (note: if doing this rename
    // the function)
    pub fn in_circle_aabr(
        &self,
        center: Vec2<f32>,
        radius: f32,
    ) -> impl Iterator<Item = specs::Entity> + '_ {
        let center = center.map(|e| e as i32);
        let radius = radius.ceil() as i32;
        // From conversion of center above
        const CENTER_TRUNCATION_ERROR: i32 = 1;
        let max_dist = radius + CENTER_TRUNCATION_ERROR;

        let aabr = Aabr {
            min: center - max_dist,
            max: center + max_dist,
        };

        self.in_aabr(aabr)
    }

    pub fn into_inner(self) -> SpatialGrid {
        SpatialGridInner {
            grid: self.grid.into_inner(),
            large_grid: self.large_grid.into_inner(),
            lg2_cell_size: self.lg2_cell_size,
            lg2_large_cell_size: self.lg2_large_cell_size,
            radius_cutoff: self.radius_cutoff,
            largest_large_radius: self.largest_large_radius.into(),
        }
    }
}
