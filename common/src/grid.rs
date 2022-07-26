use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};
use vek::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grid<T> {
    cells: Vec<T>,
    size: Vec2<i32>, // TODO: use u32
}

pub trait RowGen<'a>/*<'a, 'b: 'a>*/ {
    type Row<'b> where Self: 'b;
    type Col;

    fn row_gen<'b>(&'b mut self, yoff: i32) -> Self::Row<'b>;
    fn col_gen<'b>(row: &mut Self::Row<'b>, xoff: i32) -> Self::Col where Self: 'b;
}

impl<T> Grid<T> {
    pub fn from_raw(size: Vec2<i32>, raw: impl Into<Vec<T>>) -> Self {
        let cells = raw.into();
        assert_eq!(size.product() as usize, cells.len());
        Self { cells, size }
    }

    pub fn populate_from(size: Vec2<i32>, mut f: impl FnMut(Vec2<i32>) -> T) -> Self {
        Self {
            cells: (0..size.y)
                .flat_map(|y| (0..size.x).map(move |x| Vec2::new(x, y)))
                .map(&mut f)
                .collect(),
            size,
        }
    }

    #[inline]
    pub fn populate_by_row<'c, Gen/*, Row*//*, Col*/, const X: u32, const Y: u32>(
        gen: &mut Gen,
        /* mut row_gen: Row, */
        /* mut col_gen: Col, */
    ) -> Self
        where
            T: Clone + Default,
            /* for<'a> Gen: RowGen<'a, 'c>, */
            Gen: RowGen<'c, Col=T>,
            /* for<'a> Row: FnMut(&'a mut Gen, i32) -> <Gen as RowGen<'c>/*<'a, 'c>*/>::Row<'a>, */
            // for<'a, 'b> Col: FnMut(&'b mut <Gen as RowGen<'c>/*<'a, 'c>*/>::Row<'a>, i32) -> T,
            [(); {X as usize}]:
    {
        let mut cells = vec![T::default(); {X as usize * Y as usize}];
        cells.array_chunks_mut::<{X as usize}>().enumerate().for_each(|(y, cells)| {
            let mut row = gen.row_gen(y as i32);
            cells.iter_mut().enumerate().for_each(|(x, cell)| {
                *cell = Gen::col_gen(&mut row, x as i32);
            });
        });
        Self {
            cells,
            /* cells: (0..size.y)
                .flat_map(|y| {
                    let col = row(y);
                    (0..size.x).map(col)
                })
                /* .map(&mut f) */
                .collect(), */
            size: Vec2::new(X, Y).as_(),
        }
    }

    pub fn new(size: Vec2<i32>, default_cell: T) -> Self
    where
        T: Clone,
    {
        Self {
            cells: vec![default_cell; size.product() as usize],
            size,
        }
    }

    fn idx(&self, pos: Vec2<i32>) -> Option<usize> {
        if pos.map2(self.size, |e, sz| e >= 0 && e < sz).reduce_and() {
            Some((pos.y * self.size.x + pos.x) as usize)
        } else {
            None
        }
    }

    pub fn size(&self) -> Vec2<i32> { self.size }

    pub fn get(&self, pos: Vec2<i32>) -> Option<&T> { self.cells.get(self.idx(pos)?) }

    pub fn get_mut(&mut self, pos: Vec2<i32>) -> Option<&mut T> {
        let idx = self.idx(pos)?;
        self.cells.get_mut(idx)
    }

    pub fn set(&mut self, pos: Vec2<i32>, cell: T) -> Option<T> {
        let idx = self.idx(pos)?;
        self.cells.get_mut(idx).map(|c| core::mem::replace(c, cell))
    }

    pub fn iter(&self) -> impl Iterator<Item = (Vec2<i32>, &T)> + '_ {
        let w = self.size.x;
        self.cells
            .iter()
            .enumerate()
            .map(move |(i, cell)| (Vec2::new(i as i32 % w, i as i32 / w), cell))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Vec2<i32>, &mut T)> + '_ {
        let w = self.size.x;
        self.cells
            .iter_mut()
            .enumerate()
            .map(move |(i, cell)| (Vec2::new(i as i32 % w, i as i32 / w), cell))
    }

    pub fn iter_area(
        &self,
        pos: Vec2<i32>,
        size: Vec2<i32>,
    ) -> impl Iterator<Item = (Vec2<i32>, &T)> + '_ {
        (0..size.x).flat_map(move |x| {
            (0..size.y).flat_map(move |y| {
                Some((
                    pos + Vec2::new(x, y),
                    &self.cells[self.idx(pos + Vec2::new(x, y))?],
                ))
            })
        })
    }

    pub fn raw(&self) -> &[T] { &self.cells }
}

impl<T> Index<Vec2<i32>> for Grid<T> {
    type Output = T;

    fn index(&self, index: Vec2<i32>) -> &Self::Output {
        self.get(index).unwrap_or_else(|| {
            panic!(
                "Attempted to index grid of size {:?} with index {:?}",
                self.size(),
                index
            )
        })
    }
}

impl<T> IndexMut<Vec2<i32>> for Grid<T> {
    fn index_mut(&mut self, index: Vec2<i32>) -> &mut Self::Output {
        let size = self.size();
        self.get_mut(index).unwrap_or_else(|| {
            panic!(
                "Attempted to index grid of size {:?} with index {:?}",
                size, index
            )
        })
    }
}
