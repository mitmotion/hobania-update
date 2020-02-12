pub const GIT_VERSION: &str = include_str!(concat!(env!("OUT_DIR"), "/githash"));

lazy_static::lazy_static! {
    pub static ref GIT_HASH: &'static str = include_str!(concat!(env!("OUT_DIR"), "/githash")).split("/").nth(0).expect("failed to retrieve git_hash!");
    pub static ref GIT_DATE: &'static str = include_str!(concat!(env!("OUT_DIR"), "/githash")).split("/").nth(1).expect("failed to retrieve git_date!");
}

use std::{f32, num::FpCategory};
use vek::{Mat3, Rgb, Rgba, Vec3};

const PAIR_4_TO_COMBINATION: [[i8; 16]; 16] = [
    [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    [
        0, -1, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    ],
    [
        1, 15, -1, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    ],
    [
        2, 16, 29, -1, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
    ],
    [
        3, 17, 30, 42, -1, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    ],
    [
        4, 18, 31, 43, 54, -1, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    ],
    [
        5, 19, 32, 44, 55, 65, -1, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    ],
    [
        6, 20, 33, 45, 56, 66, 75, -1, 84, 85, 86, 87, 88, 89, 90, 91,
    ],
    [
        7, 21, 34, 46, 57, 67, 76, 84, -1, 92, 93, 94, 95, 96, 97, 98,
    ],
    [
        8, 22, 35, 47, 58, 68, 77, 85, 92, -1, 99, 100, 101, 102, 103, 104,
    ],
    [
        9, 23, 36, 48, 59, 69, 78, 86, 93, 99, -1, 105, 106, 107, 108, 109,
    ],
    [
        10, 24, 37, 49, 60, 70, 79, 87, 94, 100, 105, -1, 110, 111, 112, 113,
    ],
    [
        11, 25, 38, 50, 61, 71, 80, 88, 95, 101, 106, 110, -1, 114, 115, 116,
    ],
    [
        12, 26, 39, 51, 62, 72, 81, 89, 96, 102, 107, 111, 114, -1, 117, 118,
    ],
    [
        13, 27, 40, 52, 63, 73, 82, 90, 97, 103, 108, 112, 115, 117, -1, 119,
    ],
    [
        14, 28, 41, 53, 64, 74, 83, 91, 98, 104, 109, 113, 116, 118, 119, -1,
    ],
];

const COMBINATION_TO_PAIR_4: [u8; 120] = [
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x12,
    0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x23, 0x24, 0x25,
    0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E,
    0x4F, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x67, 0x68, 0x69, 0x6A, 0x6B,
    0x6C, 0x6D, 0x6E, 0x6F, 0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F, 0x89, 0x8A, 0x8B, 0x8C,
    0x8D, 0x8E, 0x8F, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF, 0xBC, 0xBD,
    0xBE, 0xBF, 0xCD, 0xCE, 0xCF, 0xDE, 0xDF, 0xEF,
];

/// Function converting a pair of distinct 4-bit values to an encoded
/// combinatorial index.  If values with more than 4 bits are given, only the
/// least significant 4 bits will be used.
///
/// Returns Ok(u8) on success, and Err(()) if the two values in the pair were
/// equal modulo 16.
pub fn pair_4_to_combination(a: u8, b: u8) -> Result<u8, ()> {
    let res = PAIR_4_TO_COMBINATION[(a as usize) & 15][(b as usize) & 15];
    if res < 0 { Err(()) } else { Ok(res as u8) }
}

/// Function decoding a pair of distinct 4-bit values from an encoded
/// combinatorial index.  The returned values are always in order from lowest to
/// highest.
///
/// Returns Ok(u8) on success, and Err(()) if the value provided is not a valid
/// encoded combination (valid combinations take values from 0 to 119).
pub fn combination_to_pair_4(combination: u8) -> Result<(u8, u8), ()> {
    let index = combination as usize;
    if index >= COMBINATION_TO_PAIR_4.len() {
        Err(())
    } else {
        let res = COMBINATION_TO_PAIR_4[index];
        Ok((res >> 4, res & 15))
    }
}

const F7_BIAS: i32 = 9; //2
const F7_MANTISSA: u32 = 3; //4
const F7_EXP: u32 = 7 - F7_MANTISSA;
const F7_MAX_EXP: u8 = (1 << F7_EXP) - 1;

/// Function converting a *non-negative* f32 to an f7
/// (0.F7_EXP.F7_MANTISSA.F7_BIAS minifloat) using round-to-nearest-even.
///
/// The rounding algorithm may not be entirely correct when applied to subnormal
/// numbers (it may suffer a bit from double rounding).  For now, we consider
/// this acceptable until proven otherwise.
///
/// Currently, we do round to Inf on overflow, and do have NaN representations.
/// That may not be desired, so the function may be revised to use the Inf / NaN
/// bits for other purposes (possibly removing them entirely), and to error when
/// these are provided as values (or at least, when NaNs are provided as
/// vlaues).  For now, we preserve them to try to stay as close as possible to
/// IEEE floating point semantics.
///
/// Returns Ok(u8) on success (number was a positive number or zero), and
/// Err(()) otherwise.
pub fn f32_to_f7(x: f32) -> Result<u8, ()> {
    if x.is_sign_negative() {
        // We lack a sign bit.
        return Err(());
    }
    match x.classify() {
        // Arbitrarily pick a NaN.
        FpCategory::Nan => Ok((F7_MAX_EXP << F7_MANTISSA) + 1),
        FpCategory::Infinite => Ok(F7_MAX_EXP << F7_MANTISSA),
        FpCategory::Zero => Ok(0x00),
        // All positive subnormal f32s round to 0 in f7.
        FpCategory::Subnormal => Ok(0x00),
        FpCategory::Normal => {
            // For our purposes (we don't send it over the wire, x is normal), to_bits() has
            // no downsides.
            let u = x.to_bits();
            // Try to use the same exponent if possible, after correcting for bias.
            // Note that we don't need to correct for sign because the number is positive.
            let exp = ((u >> (f32::MANTISSA_DIGITS - 1)) as i32) + F7_BIAS - f32::MAX_EXP;
            // println!("{:?} as {:x}; MANTISSA_DIGITS = {:?} exp = {:?}", x, u,
            // (f32::MANTISSA_DIGITS - 1), exp);
            if exp <= 0 {
                // underflow (target is subnormal).
                // NOTE: Probably not correct in all cases, but good enough for now.
                Ok(((x * ((1 << (F7_BIAS + F7_MANTISSA as i32 - 1)) as f32)).round()) as u8)
            } else if exp >= F7_MAX_EXP as i32 {
                // overflow (target is infinite).
                Ok(0x40)
            } else {
                // target is also (probably) normal.
                let significand = u & ((1 << (f32::MANTISSA_DIGITS - 1)) - 1);
                let number = (exp as u32) << F7_MANTISSA
                    | (significand >> ((f32::MANTISSA_DIGITS - 1) - F7_MANTISSA));
                // For round-to-nearest-two, we just need to consider three cases.
                const HALF: u32 = 1 << ((f32::MANTISSA_DIGITS - 1) - 1);
                Ok(if significand < HALF {
                    // Round down.
                    number
                } else if significand > HALF {
                    // Round up
                    number + 1
                } else {
                    // Exactly in the middle, round to nearest even.
                    if number & 1 == 0 { number } else { number + 1 }
                } as u8)
            }
        },
    }
}

/// Convert an encoded non-negative f7 (0.F7_EXP.F7_MANTISSA.F7_BIAS minifloat)
/// to an f32.  All numbers can be represented exactly, so this function doesn't
/// have an error condition.  The most significant bit of the input is
/// discarded.
pub fn f7_to_f32(u: u8) -> f32 {
    // Ignore the MSB.
    let u = u & 127;
    let exp = u >> F7_MANTISSA;
    let significand = u & ((1 << F7_MANTISSA) - 1);
    let (exp, significand) = if exp == 0 {
        if significand == 0 {
            // Zero.
            return 0.0;
        }
        // Subnormal.  Find first set bit, shift the significand, and adjust exponent.
        // NOTE: will always be non-negative since we know the significand has at most
        // F7_MANTISSA bits set, and will be at most 2 since we already handled
        // the zero case.
        let shift_by = significand.leading_zeros() - (8 - F7_MANTISSA);
        // Shift by this power of two to get the proper exponent.
        let exp = ((f32::MAX_EXP - F7_BIAS) - (shift_by as i32)) as u32;
        // The significand is the remaining bits, shifted left by the amount that was
        // reduced from the exponent.
        let significand = (significand & ((1 << shift_by) - 1)) << shift_by;
        (exp, significand)
    } else if exp == F7_MAX_EXP {
        // Infinite or NaN
        (f32::MAX_EXP as u32, significand)
    } else {
        // Normal, so just correct for bias.
        (((exp as i32) + f32::MAX_EXP - F7_BIAS) as u32, significand)
    };

    let number = ((exp as u32) << (f32::MANTISSA_DIGITS - 1))
        | ((significand as u32) << ((f32::MANTISSA_DIGITS - 1) - F7_MANTISSA));
    // Only manipulating on same architecture that produces the bits, so no
    // portability issues.
    f32::from_bits(number)
}

#[inline(always)]
pub fn srgb_to_linear(col: Rgb<f32>) -> Rgb<f32> {
    col.map(|c| {
        if c <= 0.104 {
            c * 0.08677088
        } else {
            0.012522878 * c + 0.682171111 * c * c + 0.305306011 * c * c * c
        }
    })
}
#[inline(always)]
pub fn linear_to_srgb(col: Rgb<f32>) -> Rgb<f32> {
    col.map(|c| {
        if c <= 0.0060 {
            c * 11.500726
        } else {
            let s1 = c.sqrt();
            let s2 = s1.sqrt();
            let s3 = s2.sqrt();
            0.585122381 * s1 + 0.783140355 * s2 - 0.368262736 * s3
        }
    })
}
#[inline(always)]
pub fn srgba_to_linear(col: Rgba<f32>) -> Rgba<f32> {
    Rgba::from_translucent(srgb_to_linear(Rgb::from(col)), col.a)
}
#[inline(always)]
pub fn linear_to_srgba(col: Rgba<f32>) -> Rgba<f32> {
    Rgba::from_translucent(linear_to_srgb(Rgb::from(col)), col.a)
}

/// Convert rgb to hsv. Expects rgb to be [0, 1].
#[inline(always)]
pub fn rgb_to_hsv(rgb: Rgb<f32>) -> Vec3<f32> {
    let (r, g, b) = rgb.into_tuple();
    let (max, min, diff, add) = {
        let (max, min, diff, add) = if r > g {
            (r, g, g - b, 0.0)
        } else {
            (g, r, b - r, 2.0)
        };
        if b > max {
            (b, min, r - g, 4.0)
        } else {
            (max, b.min(min), diff, add)
        }
    };

    let v = max;
    let h = if max == min {
        0.0
    } else {
        let mut h = 60.0 * (add + diff / (max - min));
        if h < 0.0 {
            h += 360.0;
        }
        h
    };
    let s = if max == 0.0 { 0.0 } else { (max - min) / max };

    Vec3::new(h, s, v)
}
/// Convert hsv to rgb. Expects h [0, 360], s [0, 1], v [0, 1]
#[inline(always)]
pub fn hsv_to_rgb(hsv: Vec3<f32>) -> Rgb<f32> {
    let (h, s, v) = hsv.into_tuple();
    let c = s * v;
    let h = h / 60.0;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h >= 0.0 && h <= 1.0 {
        (c, x, 0.0)
    } else if h <= 2.0 {
        (x, c, 0.0)
    } else if h <= 3.0 {
        (0.0, c, x)
    } else if h <= 4.0 {
        (0.0, x, c)
    } else if h <= 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    Rgb::new(r + m, g + m, b + m)
}
/// Convert linear rgb to CIExyY
#[inline(always)]
pub fn rgb_to_xyy(rgb: Rgb<f32>) -> Vec3<f32> {
    // XYZ
    let xyz = Mat3::new(
        0.4124, 0.3576, 0.1805, 0.2126, 0.7152, 0.0722, 0.0193, 0.1192, 0.9504,
    ) * Vec3::from(rgb);

    let sum = xyz.sum();
    Vec3::new(xyz.x / sum, xyz.y / sum, xyz.y)
}
/// Convert to CIExyY to linear rgb
#[inline(always)]
pub fn xyy_to_rgb(xyy: Vec3<f32>) -> Rgb<f32> {
    let xyz = Vec3::new(
        xyy.z / xyy.y * xyy.x,
        xyy.z,
        xyy.z / xyy.y * (1.0 - xyy.x - xyy.y),
    );

    Rgb::from(
        Mat3::new(
            3.2406, -1.5372, -0.4986, -0.9689, 1.8758, 0.0415, 0.0557, -0.2040, 1.0570,
        ) * xyz,
    )
}

// TO-DO: speed this up
#[inline(always)]
pub fn saturate_srgb(col: Rgb<f32>, value: f32) -> Rgb<f32> {
    let mut hsv = rgb_to_hsv(srgb_to_linear(col));
    hsv.y *= 1.0 + value;
    linear_to_srgb(hsv_to_rgb(hsv).map(|e| e.min(1.0).max(0.0)))
}

/// Preserves the luma of one color while changing its chromaticty to match the
/// other
#[inline(always)]
pub fn chromify_srgb(luma: Rgb<f32>, chroma: Rgb<f32>) -> Rgb<f32> {
    let l = rgb_to_xyy(srgb_to_linear(luma)).z;
    let mut xyy = rgb_to_xyy(srgb_to_linear(chroma));
    xyy.z = l;

    linear_to_srgb(xyy_to_rgb(xyy).map(|e| e.min(1.0).max(0.0)))
}
