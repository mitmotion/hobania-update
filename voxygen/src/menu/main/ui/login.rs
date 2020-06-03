use super::{IcedImgs as Imgs, Info, LoginInfo, Message};
use crate::{
    i18n::Localization,
    ui::{
        fonts::IcedFonts as Fonts,
        ice::{
            component::neat_button,
            style,
            widget::{
                compound_graphic::{CompoundGraphic, Graphic},
                BackgroundContainer, Image, Padding,
            },
            Element,
        },
    },
};
use iced::{
    button, text_input, Align, Column, Container, HorizontalAlignment, Length, Row, Space, Text,
    TextInput,
};
use vek::*;

const TEXT_COLOR: iced::Color = iced::Color::from_rgb(1.0, 1.0, 1.0);
const DISABLED_TEXT_COLOR: iced::Color = iced::Color::from_rgba(1.0, 1.0, 1.0, 0.2);
const FILL_FRAC_ONE: f32 = 0.77;
const FILL_FRAC_TWO: f32 = 0.53;
const INPUT_WIDTH: u16 = 250;
const INPUT_TEXT_SIZE: u16 = 24;

/// Login screen for the main menu
pub struct Screen {
    quit_button: button::State,
    settings_button: button::State,
    servers_button: button::State,

    pub banner: Banner,
}

impl Screen {
    pub fn new() -> Self {
        Self {
            servers_button: Default::default(),
            settings_button: Default::default(),
            quit_button: Default::default(),

            banner: Banner::new(),
        }
    }

    pub(super) fn view(
        &mut self,
        fonts: &Fonts,
        imgs: &Imgs,
        login_info: &LoginInfo,
        info: &Info,
        error: Option<&str>,
        version: &str,
        show_servers: bool,
        i18n: &Localization,
    ) -> Element<Message> {
        let button_style = style::button::Style::new(imgs.button)
            .hover_image(imgs.button_hover)
            .press_image(imgs.button_press)
            .text_color(TEXT_COLOR)
            .disabled_text_color(DISABLED_TEXT_COLOR);

        let buttons = Column::with_children(vec![
            neat_button(
                &mut self.servers_button,
                i18n.get("common.servers"),
                FILL_FRAC_ONE,
                button_style,
                Some(Message::ShowServers),
            ),
            neat_button(
                &mut self.settings_button,
                i18n.get("common.settings"),
                FILL_FRAC_ONE,
                button_style,
                None,
            ),
            neat_button(
                &mut self.quit_button,
                i18n.get("common.quit"),
                FILL_FRAC_ONE,
                button_style,
                Some(Message::Quit),
            ),
        ])
        .width(Length::Fill)
        .max_width(200)
        .spacing(5);

        let buttons = Container::new(buttons)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_y(Align::End);

        let left_column = if matches!(info, Info::Intro) {
            let intro_text = i18n.get("main.login_process");

            let info_window = BackgroundContainer::new(
                CompoundGraphic::from_graphics(vec![
                    Graphic::rect(Rgba::new(0, 0, 0, 240), [500, 300], [0, 0]),
                    // Note: a way to tell it to keep the height of this one piece constant and
                    // unstreched would be nice, I suppose we could just break this out into a
                    // column and use Length::Units
                    Graphic::image(imgs.banner_bottom, [500, 30], [0, 300])
                        .color(Rgba::new(255, 255, 255, 240)),
                ])
                .height(Length::Shrink),
                Text::new(intro_text).size(fonts.cyri.scale(21)),
            )
            .max_width(450)
            .padding(Padding::new().horizontal(20).top(10).bottom(60));

            Column::with_children(vec![info_window.into(), buttons.into()])
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(27)
                .into()
        } else {
            buttons.into()
        };

        let banner = self
            .banner
            .view(fonts, imgs, login_info, i18n, button_style);

        let central_column = Container::new(banner)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Align::Center)
            .align_y(Align::Center);

        let right_column = Text::new(version)
            .size(fonts.cyri.scale(15))
            .width(Length::Fill)
            .horizontal_alignment(HorizontalAlignment::Right);

        let content = Row::with_children(vec![
            left_column,
            central_column.into(),
            right_column.into(),
        ])
        .width(Length::Fill)
        .height(Length::Fill)
        .spacing(10)
        .padding(3);

        Container::new(content)
            .style(style::container::Style::image(imgs.bg))
            .into()
    }
}

pub struct Banner {
    pub username: text_input::State,
    pub password: text_input::State,
    pub server: text_input::State,

    multiplayer_button: button::State,
    #[cfg(feature = "singleplayer")]
    singleplayer_button: button::State,
}

impl Banner {
    fn new() -> Self {
        Self {
            username: Default::default(),
            password: Default::default(),
            server: Default::default(),

            multiplayer_button: Default::default(),
            #[cfg(feature = "singleplayer")]
            singleplayer_button: Default::default(),
        }
    }

    fn view(
        &mut self,
        fonts: &Fonts,
        imgs: &Imgs,
        login_info: &LoginInfo,
        i18n: &Localization,
        button_style: style::button::Style,
    ) -> Element<Message> {
        let input_text_size = fonts.cyri.scale(INPUT_TEXT_SIZE);

        let banner_content = Column::with_children(vec![
            Image::new(imgs.v_logo)
                .fix_aspect_ratio()
                .height(Length::FillPortion(20))
                .into(),
            Space::new(Length::Fill, Length::FillPortion(5)).into(),
            Column::with_children(vec![
                BackgroundContainer::new(
                    Image::new(imgs.input_bg)
                        .width(Length::Units(INPUT_WIDTH))
                        .fix_aspect_ratio(),
                    TextInput::new(
                        &mut self.username,
                        "Username",
                        &login_info.username,
                        Message::Username,
                    )
                    .size(input_text_size)
                    .on_submit(Message::FocusPassword),
                )
                .padding(Padding::new().horizontal(10).top(10))
                .into(),
                BackgroundContainer::new(
                    Image::new(imgs.input_bg)
                        .width(Length::Units(INPUT_WIDTH))
                        .fix_aspect_ratio(),
                    TextInput::new(
                        &mut self.password,
                        "Password",
                        &login_info.password,
                        Message::Password,
                    )
                    .size(input_text_size)
                    .password()
                    .on_submit(Message::Multiplayer),
                )
                .padding(Padding::new().horizontal(10).top(8))
                .into(),
                BackgroundContainer::new(
                    Image::new(imgs.input_bg)
                        .width(Length::Units(INPUT_WIDTH))
                        .fix_aspect_ratio(),
                    TextInput::new(
                        &mut self.server,
                        "Server",
                        &login_info.server,
                        Message::Server,
                    )
                    .size(input_text_size)
                    .on_submit(Message::Multiplayer),
                )
                .padding(Padding::new().horizontal(10).top(8))
                .into(),
            ])
            .spacing(2)
            .height(Length::FillPortion(50))
            .into(),
            Column::with_children(vec![
                neat_button(
                    &mut self.multiplayer_button,
                    i18n.get("common.multiplayer"),
                    FILL_FRAC_TWO,
                    button_style,
                    Some(Message::Multiplayer),
                ),
                #[cfg(feature = "singleplayer")]
                neat_button(
                    &mut self.singleplayer_button,
                    i18n.get("common.singleplayer"),
                    FILL_FRAC_TWO,
                    button_style,
                    Some(Message::Singleplayer),
                ),
            ])
            .max_width(240)
            .spacing(8)
            .into(),
        ])
        .width(Length::Fill)
        .height(Length::Fill)
        .align_items(Align::Center);

        let banner = BackgroundContainer::new(
            CompoundGraphic::from_graphics(vec![
                Graphic::image(imgs.banner_top, [138, 17], [0, 0]),
                Graphic::rect(Rgba::new(0, 0, 0, 230), [130, 195], [4, 17]),
                Graphic::image(imgs.banner, [130, 15], [4, 212])
                    .color(Rgba::new(255, 255, 255, 230)),
            ])
            .fix_aspect_ratio()
            .height(Length::Fill),
            banner_content,
        )
        .padding(Padding::new().horizontal(16).vertical(20))
        .max_width(330);

        banner.into()
    }
}
