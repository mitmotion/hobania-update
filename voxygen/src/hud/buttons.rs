use super::{
    img_ids::{Imgs, ImgsRot},
    BLACK, CRITICAL_HP_COLOR, LOW_HP_COLOR, QUALITY_LEGENDARY, TEXT_COLOR,
};
use crate::{
    game_input::GameInput,
    hud::animation::animation_timer,
    ui::{fonts::Fonts, ImageFrame, Tooltip, TooltipManager, Tooltipable},
    window::KeyMouse,
    GlobalState,
};
use client::Client;
use common::comp::{SkillSet, Stats};
use conrod_core::{
    widget::{self, Button, Image, Text, UpdateArgs},
    widget_ids, Color, Colorable, Positionable, Sizeable, UiCell, Widget, WidgetCommon,
};
use i18n::Localization;
widget_ids! {
    struct Ids {
        bag,
        bag_text,
        bag_text_bg,
        bag_space,
        bag_space_bg,
        bag_show_map,
        map_button,
        map_text,
        map_text_bg,
        settings_button,
        settings_text,
        settings_text_bg,
        social_button,
        social_button_bg,
        social_text,
        social_text_bg,
        spellbook_button,
        spellbook_button_bg,
        spellbook_text,
        spellbook_text_bg,
        crafting_button,
        crafting_button_bg,
        crafting_text,
        crafting_text_bg,
        group_button,
        sp_arrow,
        sp_arrow_txt_bg,
        sp_arrow_txt,
    }
}
#[derive(WidgetCommon)]
pub struct Buttons<'a> {
    client: &'a Client,
    show_bag: bool,
    imgs: &'a Imgs,
    fonts: &'a Fonts,
    #[conrod(common_builder)]
    common: widget::CommonBuilder,
    global_state: &'a GlobalState,
    rot_imgs: &'a ImgsRot,
    tooltip_manager: &'a mut TooltipManager,
    localized_strings: &'a Localization,
    stats: &'a Stats,
    skill_set: &'a SkillSet,
    pulse: f32,
}

impl<'a> Buttons<'a> {
    pub fn new(
        client: &'a Client,
        show_bag: bool,
        imgs: &'a Imgs,
        fonts: &'a Fonts,
        global_state: &'a GlobalState,
        rot_imgs: &'a ImgsRot,
        tooltip_manager: &'a mut TooltipManager,
        localized_strings: &'a Localization,
        stats: &'a Stats,
        skill_set: &'a SkillSet,
        pulse: f32,
    ) -> Self {
        Self {
            client,
            show_bag,
            imgs,
            fonts,
            common: widget::CommonBuilder::default(),
            global_state,
            rot_imgs,
            tooltip_manager,
            localized_strings,
            stats,
            skill_set,
            pulse,
        }
    }
}

pub struct State {
    ids: Ids,
}

#[allow(clippy::enum_variant_names)] //think about renaming to ToogleEvent
pub enum Event {
    ToggleBag,
    ToggleSettings,
    ToggleMap,
    ToggleSocial,
    ToggleSpell,
    ToggleCrafting,
}

impl<'a> Widget for Buttons<'a> {
    type Event = Option<Event>;
    type State = State;
    type Style = ();

    fn init_state(&self, id_gen: widget::id::Generator) -> Self::State {
        State {
            ids: Ids::new(id_gen),
        }
    }

    fn style(&self) -> Self::Style {}

    fn update(self, args: UpdateArgs<Self>) -> Self::Event {
        common_base::prof_span!("Buttons::update");
        let UpdateArgs { state, ui, .. } = args;
        let localized_strings = self.localized_strings;

        let button_tooltip = Tooltip::new({
            // Edge images [t, b, r, l]
            // Corner images [tr, tl, br, bl]
            let edge = &self.rot_imgs.tt_side;
            let corner = &self.rot_imgs.tt_corner;
            ImageFrame::new(
                [edge.cw180, edge.none, edge.cw270, edge.cw90],
                [corner.none, corner.cw270, corner.cw90, corner.cw180],
                Color::Rgba(0.08, 0.07, 0.04, 1.0),
                5.0,
            )
        })
        .title_font_size(self.fonts.cyri.scale(15))
        .parent(ui.window)
        .desc_font_size(self.fonts.cyri.scale(12))
        .font_id(self.fonts.cyri.conrod_id)
        .desc_text_color(TEXT_COLOR);
        // Bag
        if Button::image(if !self.show_bag {
            self.imgs.bag
        } else {
            self.imgs.bag_open
        })
        .bottom_right_with_margins_on(ui.window, 5.0, 5.0)
        .hover_image(if !self.show_bag {
            self.imgs.bag_hover
        } else {
            self.imgs.bag_open_hover
        })
        .press_image(if !self.show_bag {
            self.imgs.bag_press
        } else {
            self.imgs.bag_open_press
        })
        .w_h(420.0 / 10.0, 480.0 / 10.0)
        .with_tooltip(
            self.tooltip_manager,
            &localized_strings
                .get("hud.bag.inventory")
                .replace("{playername}", &self.stats.name),
            "",
            &button_tooltip,
            TEXT_COLOR,
        )
        .set(state.ids.bag, ui)
        .was_clicked()
        {
            return Some(Event::ToggleBag);
        };
        if let Some(bag) = &self
            .global_state
            .settings
            .controls
            .get_binding(GameInput::Bag)
        {
            self.create_new_button_with_shadow(
                ui,
                bag,
                state.ids.bag,
                state.ids.bag_text_bg,
                state.ids.bag_text,
            );
        }
        let invs = self.client.inventories();
        let inventory = match invs.get(self.client.entity()) {
            Some(inv) => inv,
            None => return None,
        };
        if !self.show_bag {
            let space_used = inventory.populated_slots();
            let space_max = inventory.slots().count();
            let bag_space = format!("{}/{}", space_used, space_max);
            let bag_space_percentage = space_used as f32 / space_max as f32;
            Text::new(&bag_space)
                .mid_top_with_margin_on(state.ids.bag, -15.0)
                .font_size(12)
                .font_id(self.fonts.cyri.conrod_id)
                .color(BLACK)
                .set(state.ids.bag_space_bg, ui);
            Text::new(&bag_space)
                .top_left_with_margins_on(state.ids.bag_space_bg, -1.0, -1.0)
                .font_size(12)
                .font_id(self.fonts.cyri.conrod_id)
                .color(if bag_space_percentage < 0.8 {
                    TEXT_COLOR
                } else if bag_space_percentage < 1.0 {
                    LOW_HP_COLOR
                } else {
                    CRITICAL_HP_COLOR
                })
                .set(state.ids.bag_space, ui);
        }
        // Settings
        if Button::image(self.imgs.settings)
            .w_h(29.0, 25.0)
            .bottom_right_with_margins_on(ui.window, 5.0, 57.0)
            .hover_image(self.imgs.settings_hover)
            .press_image(self.imgs.settings_press)
            .with_tooltip(
                self.tooltip_manager,
                localized_strings.get("common.settings"),
                "",
                &button_tooltip,
                TEXT_COLOR,
            )
            .set(state.ids.settings_button, ui)
            .was_clicked()
        {
            return Some(Event::ToggleSettings);
        };
        if let Some(settings) = &self
            .global_state
            .settings
            .controls
            .get_binding(GameInput::Settings)
        {
            self.create_new_button_with_shadow(
                ui,
                settings,
                state.ids.settings_button,
                state.ids.settings_text_bg,
                state.ids.settings_text,
            );
        };

        // Social
        if Button::image(self.imgs.social)
            .w_h(25.0, 25.0)
            .left_from(state.ids.settings_button, 10.0)
            .hover_image(self.imgs.social_hover)
            .press_image(self.imgs.social_press)
            .with_tooltip(
                self.tooltip_manager,
                localized_strings.get("hud.social"),
                "",
                &button_tooltip,
                TEXT_COLOR,
            )
            .set(state.ids.social_button, ui)
            .was_clicked()
        {
            return Some(Event::ToggleSocial);
        }
        if let Some(social) = &self
            .global_state
            .settings
            .controls
            .get_binding(GameInput::Social)
        {
            self.create_new_button_with_shadow(
                ui,
                social,
                state.ids.social_button,
                state.ids.social_text_bg,
                state.ids.social_text,
            );
        };
        // Map
        if Button::image(self.imgs.map_button)
            .w_h(22.0, 25.0)
            .left_from(state.ids.social_button, 10.0)
            .hover_image(self.imgs.map_hover)
            .press_image(self.imgs.map_press)
            .with_tooltip(
                self.tooltip_manager,
                localized_strings.get("hud.map.map_title"),
                "",
                &button_tooltip,
                TEXT_COLOR,
            )
            .set(state.ids.map_button, ui)
            .was_clicked()
        {
            return Some(Event::ToggleMap);
        };
        if let Some(map) = &self
            .global_state
            .settings
            .controls
            .get_binding(GameInput::Map)
        {
            self.create_new_button_with_shadow(
                ui,
                map,
                state.ids.map_button,
                state.ids.map_text_bg,
                state.ids.map_text,
            );
        }
        // Diary
        let unspent_sp = self.skill_set.has_available_sp();
        if Button::image(if !unspent_sp {
            self.imgs.spellbook_button
        } else {
            self.imgs.spellbook_hover
        })
        .w_h(28.0, 25.0)
        .left_from(state.ids.map_button, 10.0)
        .hover_image(self.imgs.spellbook_hover)
        .press_image(self.imgs.spellbook_press)
        .with_tooltip(
            self.tooltip_manager,
            localized_strings.get("hud.diary"),
            "",
            &button_tooltip,
            TEXT_COLOR,
        )
        .set(state.ids.spellbook_button, ui)
        .was_clicked()
        {
            return Some(Event::ToggleSpell);
        }
        if let Some(spell) = &self
            .global_state
            .settings
            .controls
            .get_binding(GameInput::Spellbook)
        {
            self.create_new_button_with_shadow(
                ui,
                spell,
                state.ids.spellbook_button,
                state.ids.spellbook_text_bg,
                state.ids.spellbook_text,
            );
        }
        // Unspent SP indicator
        if unspent_sp {
            let arrow_ani = animation_timer(self.pulse); //Animation timer
            Image::new(self.imgs.sp_indicator_arrow)
                .w_h(20.0, 11.0)
                .graphics_for(state.ids.spellbook_button)
                .mid_top_with_margin_on(state.ids.spellbook_button, -12.0 + arrow_ani as f64)
                .color(Some(QUALITY_LEGENDARY))
                .set(state.ids.sp_arrow, ui);
            Text::new(localized_strings.get("hud.sp_arrow_txt"))
                .mid_top_with_margin_on(state.ids.sp_arrow, -18.0)
                .graphics_for(state.ids.spellbook_button)
                .font_id(self.fonts.cyri.conrod_id)
                .font_size(self.fonts.cyri.scale(14))
                .color(BLACK)
                .set(state.ids.sp_arrow_txt_bg, ui);
            Text::new(localized_strings.get("hud.sp_arrow_txt"))
                .graphics_for(state.ids.spellbook_button)
                .bottom_right_with_margins_on(state.ids.sp_arrow_txt_bg, 1.0, 1.0)
                .font_id(self.fonts.cyri.conrod_id)
                .font_size(self.fonts.cyri.scale(14))
                .color(QUALITY_LEGENDARY)
                .set(state.ids.sp_arrow_txt, ui);
        }
        // Crafting
        if Button::image(self.imgs.crafting_icon)
            .w_h(25.0, 25.0)
            .left_from(state.ids.spellbook_button, 10.0)
            .hover_image(self.imgs.crafting_icon_hover)
            .press_image(self.imgs.crafting_icon_press)
            .with_tooltip(
                self.tooltip_manager,
                localized_strings.get("hud.crafting"),
                "",
                &button_tooltip,
                TEXT_COLOR,
            )
            .set(state.ids.crafting_button, ui)
            .was_clicked()
        {
            return Some(Event::ToggleCrafting);
        }
        if let Some(crafting) = &self
            .global_state
            .settings
            .controls
            .get_binding(GameInput::Crafting)
        {
            self.create_new_button_with_shadow(
                ui,
                crafting,
                state.ids.crafting_button,
                state.ids.crafting_text_bg,
                state.ids.crafting_text,
            );
        }

        None
    }
}

impl<'a> Buttons<'a> {
    fn create_new_button_with_shadow(
        &self,
        ui: &mut UiCell,
        key_mouse: &KeyMouse,
        button_identifier: widget::Id,
        text_background: widget::Id,
        text: widget::Id,
    ) {
        let key_layout = &self.global_state.window.key_layout;
        let key_desc = key_mouse.try_shortened(key_layout);

        //Create shadow
        Text::new(&key_desc)
            .bottom_right_with_margins_on(button_identifier, 0.0, 0.0)
            .font_size(10)
            .font_id(self.fonts.cyri.conrod_id)
            .color(BLACK)
            .set(text_background, ui);

        //Create button
        Text::new(&key_desc)
            .bottom_right_with_margins_on(text_background, 1.0, 1.0)
            .font_size(10)
            .font_id(self.fonts.cyri.conrod_id)
            .color(TEXT_COLOR)
            .set(text, ui);
    }
}
