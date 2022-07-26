pub trait Sampler<'a, 'b>: Sized {
    type Index: 'a;
    type Sample: 'a;

    fn get(&'b self, index: Self::Index) -> Self::Sample;
}

pub trait SamplerMut<'a, 'b>: Sized {
    type Index: 'a;
    type Sample: 'a;

    fn get(&'b mut self, index: Self::Index) -> Self::Sample;
}
