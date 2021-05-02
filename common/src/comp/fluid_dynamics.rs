use super::{Density, Ori, Vel};
use crate::{
    consts::{AIR_DENSITY, LAVA_DENSITY, WATER_DENSITY},
    util::{Dir, Plane, Projection},
};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use vek::*;

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LiquidKind {
    Water,
    Lava,
}

impl LiquidKind {
    /// If an entity is in multiple overlapping liquid blocks, which one takes
    /// precedence? (should be a rare edge case, since checkerboard patterns of
    /// water and lava shouldn't show up in worldgen)
    pub fn merge(self, other: LiquidKind) -> LiquidKind {
        use LiquidKind::{Lava, Water};
        match (self, other) {
            (Water, Water) => Water,
            (Water, Lava) => Lava,
            (Lava, _) => Lava,
        }
    }
}

/// Fluid medium in which the entity exists
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Fluid {
    Air {
        vel: Vel,
        elevation: f32,
    },
    Liquid {
        kind: LiquidKind,
        vel: Vel,
        depth: f32,
    },
}

impl Fluid {
    /// Specific mass
    pub fn density(&self) -> Density {
        match self {
            Self::Air { .. } => Density(AIR_DENSITY),
            Self::Liquid {
                kind: LiquidKind::Water,
                ..
            } => Density(WATER_DENSITY),
            Self::Liquid {
                kind: LiquidKind::Lava,
                ..
            } => Density(LAVA_DENSITY),
        }
    }

    /// Pressure from entity velocity
    pub fn dynamic_pressure(&self, vel: &Vel) -> f32 {
        0.5 * self.density().0 * self.relative_flow(vel).0.magnitude_squared()
    }

    /*
        pub fn static_pressure(&self) -> f32 {
            match self {
                Self::Air { elevation, .. } => Self::air_pressure(*elevation),
                Self::Water { depth, .. } => Self::water_pressure(*depth),
            }
        }

        /// Absolute static pressure of air at elevation
        pub fn air_pressure(elevation: f32) -> f32 {
            // At low altitudes above sea level, the pressure decreases by about 1.2 kPa for
            // every 100 metres.
            // https://en.wikipedia.org/wiki/Atmospheric_pressure#Altitude_variation
            ATMOSPHERE - elevation / 12.0
        }

        /// Absolute static pressure of water at depth
        pub fn water_pressure(depth: f32) -> f32 { WATER_DENSITY * GRAVITY * depth + ATMOSPHERE }
    */
    /// Velocity of fluid, if applicable
    pub fn flow_vel(&self) -> Vel {
        match self {
            Self::Air { vel, .. } => *vel,
            Self::Liquid { vel, .. } => *vel,
        }
    }

    // Very simple but useful in reducing mental overhead
    pub fn relative_flow(&self, vel: &Vel) -> Vel { Vel(self.flow_vel().0 - vel.0) }

    pub fn is_liquid(&self) -> bool { matches!(self, Fluid::Liquid { .. }) }

    pub fn elevation(&self) -> Option<f32> {
        match self {
            Fluid::Air { elevation, .. } => Some(*elevation),
            _ => None,
        }
    }

    pub fn depth(&self) -> Option<f32> {
        match self {
            Fluid::Liquid { depth, .. } => Some(*depth),
            _ => None,
        }
    }
}

impl Default for Fluid {
    fn default() -> Self {
        Self::Air {
            elevation: 0.0,
            vel: Vel::zero(),
        }
    }
}

pub trait Drag: Clone {
    /// Drag coefficient for skin friction and flow separation.
    /// (Multiplied by reference area.)
    fn parasite_drag_coefficient(&self) -> f32;

    fn drag(&self, rel_flow: &Vel, fluid_density: f32) -> Vec3<f32> {
        let v_sq = rel_flow.0.magnitude_squared();
        if v_sq < 0.25 {
            // don't bother with miniscule forces
            Vec3::zero()
        } else {
            let rel_flow_dir = Dir::new(rel_flow.0 / v_sq.sqrt());
            // All the coefficients come pre-multiplied by their reference area
            0.5 * fluid_density * v_sq * self.parasite_drag_coefficient() * *rel_flow_dir
        }
    }
}

/// An implementation of Glide is the ability for a winged body to glide in a
/// fixed-wing configuration.
///
/// NOTE: Wing (singular) implies the full span of a complete wing; a wing in
/// two parts (one on each side of a fuselage or other central body) is
/// considered a single wing.
pub trait Glide: Drag {
    fn wing_shape(&self) -> &WingShape;

    fn planform_area(&self) -> f32;

    fn ori(&self) -> &Ori;

    fn is_gliding(&self) -> bool;

    fn aerodynamic_forces(&self, rel_flow: &Vel, fluid_density: f32) -> Vec3<f32> {
        if self.is_gliding() {
            let ori = self.ori();
            let planform_area = self.planform_area();
            let parasite_drag_coefficient = self.parasite_drag_coefficient();

            let f = |flow_v: Vec3<f32>, wing_shape: &WingShape| -> Vec3<f32> {
                let v_sq = flow_v.magnitude_squared();
                if v_sq < std::f32::EPSILON {
                    Vec3::zero()
                } else {
                    let freestream_dir = Dir::new(flow_v / v_sq.sqrt());

                    let ar = wing_shape.aspect_ratio();
                    // aoa will be positive when we're pitched up and negative otherwise
                    let aoa = angle_of_attack(ori, &freestream_dir);
                    // c_l will be positive when aoa is positive (we have positive lift,
                    // producing an upward force) and negative otherwise
                    let c_l = lift_coefficient(
                        planform_area,
                        aoa,
                        wing_shape.lift_slope(),
                        wing_shape.stall_angle(),
                    );

                    let c_d = parasite_drag_coefficient + induced_drag_coefficient(ar, c_l);

                    0.5 * fluid_density
                        * v_sq
                        * (c_l * *lift_dir(ori, ar, c_l, aoa) + c_d * *freestream_dir)
                }
            };

            let wing_shape = self.wing_shape();
            let lateral = f(
                rel_flow.0.projected(&Plane::from(self.ori().look_dir())),
                &wing_shape.with_aspect_ratio(1.0 / wing_shape.aspect_ratio()),
            );
            let longitudinal = f(
                rel_flow.0.projected(&Plane::from(self.ori().right())),
                &wing_shape,
            );
            lateral + longitudinal
        } else {
            self.drag(rel_flow, fluid_density)
        }
    }
}

fn lift_dir(ori: &Ori, aspect_ratio: f32, lift_coefficient: f32, angle_of_attack: f32) -> Dir {
    // lift dir will be orthogonal to the local relative flow vector. Local relative
    // flow is the resulting vector of (relative) freestream flow + downwash
    // (created by the vortices of the wing tips)

    // induced angle of attack
    let aoa_i = lift_coefficient / (PI * aspect_ratio);
    // effective angle of attack; the aoa as seen by aerofoil after downwash
    let aoa_eff = angle_of_attack - aoa_i;

    // Angle between chord line and local relative wind is aoa_eff radians.
    // Direction of lift is perpendicular to local relative wind. At positive lift,
    // local relative wind will be below our cord line at an angle of aoa_eff. Thus
    // if we pitch down by aoa_eff radians then our chord line will be colinear with
    // local relative wind vector and our up will be the direction of lift.
    ori.pitched_down(aoa_eff).up()
}

/// Geometric angle of attack
pub fn angle_of_attack(ori: &Ori, rel_flow_dir: &Dir) -> f32 {
    if inline_tweak::tweak!(true) {
        PI / 2.0 - ori.up().angle_between(rel_flow_dir.to_vec())
    } else {
        rel_flow_dir
            .projected(&Plane::from(ori.right()))
            .map(|flow_dir| PI / 2.0 - ori.up().angle_between(flow_dir.to_vec()))
            .unwrap_or(0.0)
    }
}

/// Total lift coefficient for a finite wing of symmetric aerofoil shape and
/// elliptical pressure distribution. (Multiplied by reference area.)
fn lift_coefficient(
    planform_area: f32,
    angle_of_attack: f32,
    lift_slope: f32,
    stall_angle: f32,
) -> f32 {
    let aoa_abs = angle_of_attack.abs();
    planform_area
        * if aoa_abs <= stall_angle {
            lift_slope * angle_of_attack
        } else {
            // This is when flow separation and turbulence starts to kick in.
            // Going to just make something up (based on some data), as the alternative is
            // to just throw your hands up and return 0
            let aoa_s = angle_of_attack.signum();
            let c_l_max = lift_slope * stall_angle;
            let deg_45 = PI / 4.0;
            if aoa_abs < deg_45 {
                // drop directly to 0.6 * max lift at stall angle
                // then climb back to max at 45°
                Lerp::lerp(0.6 * c_l_max, c_l_max, aoa_abs / deg_45) * aoa_s
            } else {
                // let's just say lift goes down linearly again until we're at 90°
                Lerp::lerp(c_l_max, 0.0, (aoa_abs - deg_45) / deg_45) * aoa_s
            }
        }
}

#[derive(Copy, Clone)]
pub enum WingState {
    Fixed,
    Flapping,
    Retracted,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WingShape {
    Elliptical { aspect_ratio: f32 },
    // Tapered { aspect_ratio: f32, e: f32 },
    Swept { aspect_ratio: f32, angle: f32 },
    // Delta,
}

impl WingShape {
    pub const fn stall_angle(&self) -> f32 {
        // PI/10 or 18°
        0.3141592653589793
    }

    pub const fn with_aspect_ratio(self, aspect_ratio: f32) -> Self {
        match self {
            Self::Elliptical { .. } => Self::Elliptical { aspect_ratio },
            Self::Swept { angle, .. } => Self::Swept {
                aspect_ratio,
                angle,
            },
        }
    }

    pub const fn aspect_ratio(&self) -> f32 {
        match self {
            Self::Elliptical { aspect_ratio } => *aspect_ratio,
            Self::Swept { aspect_ratio, .. } => *aspect_ratio,
        }
    }

    pub fn elliptical_planform_area(span_length: f32, chord_length: f32) -> f32 {
        std::f32::consts::PI * span_length * chord_length * 0.25
    }

    pub fn elliptical(span_length: f32, chord_length: f32) -> Self {
        Self::Elliptical {
            aspect_ratio: span_length.powi(2)
                / Self::elliptical_planform_area(span_length, chord_length),
        }
    }

    /// The change in lift over change in angle of attack¹. Multiplying by angle
    /// of attack gives the lift coefficient (for a finite wing, not aerofoil).
    /// Aspect ratio is the ratio of total wing span squared over planform area.
    ///
    /// # Notes
    /// Only valid for symmetric, elliptical wings at small² angles of attack³.
    /// Does not apply to twisted, cambered or delta wings. (It still gives a
    /// reasonably accurate approximation if the wing shape is not truly
    /// elliptical.)
    /// 1. geometric angle of attack, i.e. the pitch angle relative to
    /// freestream flow
    /// 2. up to around ~18°, at which point maximum lift has been achieved and
    /// thereafter falls precipitously, causing a stall (this is the stall
    /// angle) 3. effective aoa, i.e. geometric aoa - induced aoa; assumes
    /// no sideslip
    pub fn lift_slope(&self) -> f32 {
        // lift slope for a thin aerofoil, given by Thin Aerofoil Theory
        let a0 = 2.0 * PI;
        match self {
            WingShape::Elliptical { aspect_ratio } => {
                if aspect_ratio < &4.0 {
                    // for low aspect ratio wings (AR < 4) we use Helmbold's equation
                    let x = a0 / (PI * aspect_ratio);
                    a0 / ((1.0 + x.powi(2)).sqrt() + x)
                } else {
                    // for high aspect ratio wings (AR > 4) we use the equation given by
                    // Prandtl's lifting-line theory
                    a0 / (1.0 + (a0 / (PI * aspect_ratio)))
                }
            },
            // WingShape::Tapered { aspect_ratio, e } => todo!(),
            WingShape::Swept {
                aspect_ratio,
                angle,
            } => {
                // for swept wings we use Kuchemann's modification to Helmbold's
                // equation
                let a0_cos_sweep = a0 * angle.cos();
                let x = a0_cos_sweep / (PI * aspect_ratio);
                a0_cos_sweep / ((1.0 + x.powi(2)).sqrt() + x)
            },
        }
    }
}

/// Induced drag coefficient (drag due to lift)
fn induced_drag_coefficient(aspect_ratio: f32, lift_coefficient: f32) -> f32 {
    let ar = aspect_ratio;
    if ar > 25.0 {
        tracing::warn!(
            "Calculating induced drag for wings with a given aspect ratio of {}. The formulas are \
             only valid for aspect ratios below 25, so it will be substituted.",
            ar
        )
    };
    let ar = ar.min(24.0);
    // Oswald's efficiency factor (empirically derived--very magical)
    // (this definition should not be used for aspect ratios > 25)
    let e = 1.78 * (1.0 - 0.045 * ar.powf(0.68)) - 0.64;
    // induced drag coefficient (drag due to lift)
    lift_coefficient.powi(2) / (PI * e * ar)
}
