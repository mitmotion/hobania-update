#include <constants.glsl>
#include <random.glsl>
#include <light.glsl>
#include <lod.glsl>

float falloff(float x) {
    return pow(max(x > 0.577 ? (0.3849 / x - 0.1) : (0.9 - x * x), 0.0), 4);
}

// Return the 'broad' density of the cloud at a position. This gets refined later with extra noise, but is important
// for computing light access.
float cloud_broad(vec3 pos) {
    return 0.0
        + 2 * (noise_3d(pos / vec3(vec2(30000.0), 20000.0) / cloud_scale + 1000.0) - 0.5)
    ;
}

// Returns vec4(r, g, b, density)
vec4 cloud_at(vec3 pos, float dist, out vec3 emission, out float not_underground) {
    #ifdef EXPERIMENTAL_CURVEDWORLD
        pos.z += pow(distance(pos.xy, focus_pos.xy + focus_off.xy) * 0.05, 2);
    #endif

    // Natural attenuation of air (air naturally attenuates light that passes through it)
    // Simulate the atmosphere thinning as you get higher. Not physically accurate, but then
    // it can't be since Veloren's world is flat, not spherical.
    float atmosphere_alt = CLOUD_AVG_ALT + 40000.0;
    // Veloren's world is flat. This is, to put it mildly, somewhat non-physical. With the earth as an infinitely-big
    // plane, the atmosphere is therefore capable of scattering 100% of any light source at the horizon, no matter how
    // bright, because it has to travel through an infinite amount of atmosphere. This doesn't happen in reality
    // because the earth has curvature and so there is an upper bound on the amount of atmosphere that a sunset must
    // travel through. We 'simulate' this by fading out the atmosphere density with distance.
    float flat_earth_hack = 1.0 / (1.0 + dist * 0.0001);
    float air = 0.025 * clamp((atmosphere_alt - pos.z) / 20000, 0, 1) * flat_earth_hack;

    float alt = alt_at(pos.xy - focus_off.xy);

    // Mist sits close to the ground in valleys (TODO: use base_alt to put it closer to water)
    float mist_min_alt = 0.5;
    #if (CLOUD_MODE >= CLOUD_MODE_MEDIUM)
        mist_min_alt = (textureLod(sampler2D(t_noise, s_noise), pos.xy / 50000.0, 0).x - 0.5) * 1.5 + 0.5;
    #endif
    mist_min_alt = view_distance.z * 1.5 * (1.0 + mist_min_alt * 0.5) + alt * 0.5 + 250;
    const float MIST_FADE_HEIGHT = 1000;
    float mist = 0.01 * pow(clamp(1.0 - (pos.z - mist_min_alt) / MIST_FADE_HEIGHT, 0.0, 1), 10.0) * flat_earth_hack;

    vec3 wind_pos = vec3(pos.xy + wind_offset, pos.z + noise_2d(pos.xy / 20000) * 500);

    // Clouds
    float cloud_tendency = cloud_tendency_at(pos.xy);
    float cloud = 0;

    if (mist > 0.0) {
        mist *= 0.5
        #if (CLOUD_MODE >= CLOUD_MODE_LOW)
            + 1.0 * (noise_2d(wind_pos.xy / 5000) - 0.5)
        #endif
        #if (CLOUD_MODE >= CLOUD_MODE_MEDIUM)
            + 0.25 * (noise_3d(wind_pos / 1000) - 0.5)
        #endif
        ;
    }

    //vec2 cloud_attr = get_cloud_heights(wind_pos.xy);
    float sun_access = 0.0;
    float moon_access = 0.0;
    float cloud_sun_access = 0.0;
    float cloud_moon_access = 0.0;
    float cloud_broad_a = 0.0;
    float cloud_broad_b = 0.0;
    // This is a silly optimisation but it actually nets us a fair few fps by skipping quite a few expensive calcs
    if ((pos.z < CLOUD_AVG_ALT + 15000.0 && cloud_tendency > 0.0)) {
        // Turbulence (small variations in clouds/mist)
        const float turb_speed = -1.0; // Turbulence goes the opposite way
        vec3 turb_offset = vec3(1, 1, 0) * time_of_day.x * turb_speed;

        float CLOUD_DEPTH = (view_distance.w - view_distance.z) * 0.8;
        const float CLOUD_DENSITY = 10000.0;
        const float CLOUD_ALT_VARI_WIDTH = 100000.0;
        const float CLOUD_ALT_VARI_SCALE = 5000.0;
        float cloud_alt = CLOUD_AVG_ALT + alt * 0.5;

        cloud_broad_a = cloud_broad(wind_pos + sun_dir.xyz * 250);
        cloud_broad_b = cloud_broad(wind_pos - sun_dir.xyz * 250);
        cloud = cloud_tendency + (0.0
            + 24 * (cloud_broad_a + cloud_broad_b) * 0.5
        #if (CLOUD_MODE >= CLOUD_MODE_MINIMAL)
            + 4 * (noise_3d((wind_pos + turb_offset) / 2000.0 / cloud_scale) - 0.5)
        #endif
        #if (CLOUD_MODE >= CLOUD_MODE_LOW)
            + 0.75 * (noise_3d((wind_pos + turb_offset * 0.5) / 750.0 / cloud_scale) - 0.5)
        #endif
        #if (CLOUD_MODE >= CLOUD_MODE_HIGH)
            + 0.75 * (noise_3d(wind_pos / 500.0 / cloud_scale) - 0.5)
        #endif
        ) * 0.01;
        cloud = pow(max(cloud, 0), 3) * sign(cloud);
        cloud *= CLOUD_DENSITY * sqrt(cloud_tendency) * falloff(abs(pos.z - cloud_alt) / CLOUD_DEPTH);

        // What proportion of sunlight is *not* being blocked by nearby cloud? (approximation)
        // Basically, just throw together a few values that roughly approximate this term and come up with an average
        cloud_sun_access = exp((
            // Cloud density gradient
            0.25 * (cloud_broad_a - cloud_broad_b + (0.25 * (noise_3d(wind_pos / 4000 / cloud_scale) - 0.5) + 0.1 * (noise_3d(wind_pos / 1000 / cloud_scale) - 0.5)))
        #if (CLOUD_MODE >= CLOUD_MODE_HIGH)
            // More noise
            + 0.01 * (noise_3d(wind_pos / 500) / cloud_scale - 0.5)
        #endif
        ) * 15.0 - 1.5) * 1.5;
        // Since we're assuming the sun/moon is always above (not always correct) it's the same for the moon
        cloud_moon_access = 1.0 - cloud_sun_access;
    }

    // Keeping this because it's something I'm likely to reenable later
    /*
    #if (CLOUD_MODE >= CLOUD_MODE_HIGH)
        // Try to calculate a reasonable approximation of the cloud normal
        float cloud_tendency_x = cloud_tendency_at(pos.xy + vec2(100, 0));
        float cloud_tendency_y = cloud_tendency_at(pos.xy + vec2(0, 100));
        vec3 cloud_norm = vec3(
            (cloud_tendency - cloud_tendency_x) * 4,
            (cloud_tendency - cloud_tendency_y) * 4,
            (pos.z - cloud_attr.x) / cloud_attr.y + 0.5
        );
        cloud_sun_access = mix(max(dot(-sun_dir.xyz, cloud_norm) - 1.0, 0.025), cloud_sun_access, 0.25);
        cloud_moon_access = mix(max(dot(-moon_dir.xyz, cloud_norm) - 0.6, 0.025), cloud_moon_access, 0.25);
    #endif
    */

    float mist_sun_access = exp(mist);
    float mist_moon_access = mist_sun_access;
    sun_access = mix(cloud_sun_access, mist_sun_access, clamp(mist * 20000, 0, 1));
    moon_access = mix(cloud_moon_access, mist_moon_access, clamp(mist * 20000, 0, 1));

    // Prevent mist (i.e: vapour beneath clouds) being accessible to the sun to avoid visual problems
    //float suppress_mist = clamp((pos.z - cloud_attr.x + cloud_attr.y) / 300, 0, 1);
    //sun_access *= suppress_mist;
    //moon_access *= suppress_mist;

    // Prevent clouds and mist appearing underground (but fade them out gently)
    not_underground = clamp(1.0 - (alt - (pos.z - focus_off.z)) / 80.0 + dist * 0.001, 0, 1);
    sun_access *= not_underground;
    moon_access *= not_underground;
    float vapor_density = (mist + cloud) * not_underground;

    if (emission_strength <= 0.0) {
        emission = vec3(0);
    } else {
        float nz = textureLod(sampler2D(t_noise, s_noise), wind_pos.xy * 0.00005 - time_of_day.x * 0.0001, 0).x;//noise_3d(vec3(wind_pos.xy * 0.00005 + cloud_tendency * 0.2, time_of_day.x * 0.0002));

        float emission_alt = alt * 0.5 + 1000 + 1000 * nz;
        float emission_height = 1000.0;
        float emission_factor = pow(max(0.0, 1.0 - abs((pos.z - emission_alt) / emission_height - 1.0))
            * max(0, 1.0 - abs(0.0
                + textureLod(sampler2D(t_noise, s_noise), wind_pos.xy * 0.0001 + nz * 0.1, 0).x
                + textureLod(sampler2D(t_noise, s_noise), wind_pos.xy * 0.0005 + nz * 0.5, 0).x * 0.3
                - 0.5) * 2)
            * max(0, 1.0 - abs(textureLod(sampler2D(t_noise, s_noise), wind_pos.xy * 0.00001, 0).x - 0.5) * 4)
            , 2) * emission_strength;
        float t = clamp((pos.z - emission_alt) / emission_height, 0, 1);
        t = pow(t - 0.5, 2) * sign(t - 0.5) + 0.5;
        float top = pow(t, 2);
        float bot = pow(max(0.8 - t, 0), 2) * 2;
        const vec3 cyan = vec3(0, 0.5, 1);
        const vec3 red = vec3(1, 0, 0);
        const vec3 green = vec3(0, 8, 0);
        emission = 10 * emission_factor * nz * (cyan * top * max(0, 1 - emission_br) + red * max(emission_br, 0) + green * bot);
    }

    // We track vapor density and air density separately. Why? Because photons will ionize particles in air
    // leading to rayleigh scattering, but water vapor will not. Tracking these indepedently allows us to
    // get more correct colours.
    return vec4(sun_access, moon_access, vapor_density, air);
}

float atan2(in float y, in float x) {
    bool s = (abs(x) > abs(y));
    return mix(PI/2.0 - atan(x,y), atan(y,x), s);
}

const float DIST_CAP = 50000;
#if (CLOUD_MODE == CLOUD_MODE_ULTRA)
    const uint QUALITY = 200u;
#elif (CLOUD_MODE == CLOUD_MODE_HIGH)
    const uint QUALITY = 40u;
#elif (CLOUD_MODE == CLOUD_MODE_MEDIUM)
    const uint QUALITY = 18u;
#elif (CLOUD_MODE == CLOUD_MODE_LOW)
    const uint QUALITY = 6u;
#elif (CLOUD_MODE == CLOUD_MODE_MINIMAL)
    const uint QUALITY = 2u;
#endif

const float STEP_SCALE = DIST_CAP / (10.0 * float(QUALITY));

float step_to_dist(float step, float quality) {
    return pow(step, 2) * STEP_SCALE / quality;
}

float dist_to_step(float dist, float quality) {
    return pow(dist / STEP_SCALE * quality, 0.5);
}

vec3 apply_point_glow(vec3 wpos, vec3 dir, float max_dist, vec3 color, const float factor) {
    #ifndef EXPERIMENTAL_NOPOINTGLOW
        for (uint i = 0u; i < light_shadow_count.x; i ++) {
            // Only access the array once
            Light L = lights[i];

            vec3 light_pos = L.light_pos.xyz;
            // Project light_pos to dir line
            float t = max(dot(light_pos - wpos, dir), 0);
            vec3 nearest = wpos + dir * min(t, max_dist);

            //if (t > max_dist) { continue; }

            vec3 difference = light_pos - nearest;
            #if (CLOUD_MODE >= CLOUD_MODE_HIGH)
                vec3 _unused;
                float unused2;
                float spread = 1.0 / (1.0 + cloud_at(nearest, 0.0, _unused, unused2).z * 0.005);
            #else
                const float spread = 1.0;
            #endif
            float distance_2 = dot(difference, difference);
            if (distance_2 > 100000.0) {
                continue;
            }

            float strength = pow(attenuation_strength_real(difference), spread);

            vec3 light_color = srgb_to_linear(L.light_col.rgb) * strength * L.light_col.a;

            const float LIGHT_AMBIANCE = 0.025;
            color += light_color
                * 0.025
                // Constant, *should* const fold
                * pow(factor, 0.65);
        }
    #endif
    return color;
}

vec3 get_cloud_color(vec3 surf_color, vec3 dir, vec3 origin, const float time_of_day, float max_dist, const float quality) {
    // Limit the marching distance to reduce maximum jumps
    max_dist = min(max_dist, DIST_CAP);

    origin.xyz += focus_off.xyz;

    // This hack adds a little direction-dependent noise to clouds. It's not correct, but it very cheaply
    // improves visual quality for low cloud settings
    float splay = 1.0;
    #if (CLOUD_MODE == CLOUD_MODE_MINIMAL)
        splay += (textureLod(sampler2D(t_noise, s_noise), vec2(atan2(dir.x, dir.y) * 2 / PI, dir.z) * 5.0 - time_of_day * 0.00005, 0).x - 0.5) * 0.025 / (1.0 + pow(dir.z, 2) * 10);
    #endif

    const vec3 RAYLEIGH = vec3(0.025, 0.1, 0.5);

    // Proportion of sunlight that get scattered back into the camera by clouds
    float sun_scatter = dot(-dir, sun_dir.xyz) * 0.5 + 0.7;
    float moon_scatter = dot(-dir, moon_dir.xyz) * 0.5 + 0.7;
    float net_light = get_sun_brightness() + get_moon_brightness();
    vec3 sky_color = RAYLEIGH * net_light;
    vec3 sky_light = get_sky_light(dir, time_of_day, false);
    vec3 sun_color = get_sun_color();
    vec3 moon_color = get_moon_color();

    float cdist = max_dist;
    float ldist = cdist;
    // i is an emergency brake
    float min_dist = clamp(max_dist / 4, 0.25, 24);
    int i;
    for (i = 0; cdist > min_dist && i < 250; i ++) {
        ldist = cdist;
        cdist = step_to_dist(trunc(dist_to_step(cdist - 0.25, quality)), quality);

        vec3 emission;
        float not_underground; // Used to prevent sunlight leaking underground
        // `sample` is a reserved keyword
        vec4 sample_ = cloud_at(origin + dir * ldist * splay, ldist, emission, not_underground);

        vec2 density_integrals = max(sample_.zw, vec2(0));

        float sun_access = max(sample_.x, 0);
        float moon_access = max(sample_.y, 0);
        float cloud_scatter_factor = density_integrals.x;
        float global_scatter_factor = density_integrals.y;

        float step = (ldist - cdist) * 0.01;
        float cloud_darken = pow(1.0 / (1.0 + cloud_scatter_factor), step);
        float global_darken = pow(1.0 / (1.0 + global_scatter_factor), step);

        surf_color =
            // Attenuate light passing through the clouds
            surf_color * cloud_darken * global_darken +
            // Add the directed light light scattered into the camera by the clouds and the atmosphere (global illumination)
            sun_color * sun_scatter * get_sun_brightness() * (sun_access * (1.0 - cloud_darken) /*+ sky_color * global_scatter_factor*/) +
            moon_color * moon_scatter * get_moon_brightness() * (moon_access * (1.0 - cloud_darken) /*+ sky_color * global_scatter_factor*/) +
            sky_light * (1.0 - global_darken) * not_underground +
            emission * density_integrals.y * step;
    }

    // Apply point glow
    #ifdef BLOOM_FACTOR
        surf_color = apply_point_glow(origin, dir, max_dist, surf_color, BLOOM_FACTOR);
    #endif

    return surf_color;
}
