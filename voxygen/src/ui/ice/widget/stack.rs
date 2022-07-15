// TODO: unused (I think?) consider slating for removal
use iced::{layout, Element, Hasher, Layout, Length, Point, Rectangle, Size, Widget};
use std::hash::Hash;

/// Stack up some widgets
pub struct Stack<'a, M, R> {
    children: Vec<Element<'a, M, R>>,
}

impl<'a, M, R> Stack<'a, M, R>
where
    R: Renderer,
{
    pub fn with_children(children: Vec<Element<'a, M, R>>) -> Self { Self { children } }
}

impl<'a, M, R> Widget<M, R> for Stack<'a, M, R>
where
    R: Renderer,
{
    fn width(&self) -> Length { Length::Fill }

    fn height(&self) -> Length { Length::Fill }

    fn layout(&self, renderer: &R, limits: &layout::Limits) -> layout::Node {
        let limits = limits.width(Length::Fill).height(Length::Fill);

        let loosed_limits = limits.loose();

        let (max_size, nodes) = self.children.iter().fold(
            (Size::ZERO, Vec::with_capacity(self.children.len())),
            |(mut max_size, mut nodes), child| {
                let node = child.layout(renderer, &loosed_limits);
                let size = node.size();
                nodes.push(node);
                max_size.width = max_size.width.max(size.width);
                max_size.height = max_size.height.max(size.height);
                (max_size, nodes)
            },
        );

        let size = limits.resolve(max_size);

        layout::Node::with_children(size, nodes)
    }

    fn draw(
        &self,
        renderer: &mut R,
        defaults: &R::Defaults,
        layout: Layout<'_>,
        cursor_position: Point,
        viewport: &Rectangle,
    ) -> R::Output {
        renderer.draw(defaults, &self.children, layout, cursor_position, viewport)
    }

    fn hash_layout(&self, state: &mut Hasher) {
        struct Marker;
        std::any::TypeId::of::<Marker>().hash(state);

        self.children
            .iter()
            .for_each(|child| child.hash_layout(state));
    }

    fn overlay(&mut self, layout: Layout<'_>) -> Option<iced::overlay::Element<'_, M, R>> {
        self.children
            .iter_mut()
            .zip(layout.children())
            .filter_map(|(child, layout)| child.overlay(layout))
            .next()
    }
}

pub trait Renderer: iced::Renderer {
    fn draw<M>(
        &mut self,
        defaults: &Self::Defaults,
        children: &[Element<'_, M, Self>],
        layout: Layout<'_>,
        cursor_position: Point,
        viewport: &Rectangle,
    ) -> Self::Output;
}

impl<'a, M, R> From<Stack<'a, M, R>> for Element<'a, M, R>
where
    R: 'a + Renderer,
    M: 'a,
{
    fn from(stack: Stack<'a, M, R>) -> Element<'a, M, R> { Element::new(stack) }
}
