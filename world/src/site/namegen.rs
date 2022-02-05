use rand::prelude::*;

pub struct NameGen<'a, R: Rng> {
    // 2..
    pub approx_syllables: usize,
    pub approx_syllables_long: usize,

    rng: &'a mut R,
}

impl<'a, R: Rng> NameGen<'a, R> {
    pub fn location(rng: &'a mut R) -> Self {
        Self {
            approx_syllables: rng.gen_range(1..4),
            approx_syllables_long: rng.gen_range(2..4),

            rng,
        }
    }

    pub fn generate(self) -> String {
        let cons = vec![
            "d", "f", "ph", "r", "st", "t", "s", "p", "sh", "th", "br", "tr", "m", "k", "st", "w",
            "y", "cr", "fr", "dr", "pl", "wr", "sn", "g", "qu", "l",
        ];
        let mut start = cons.clone();
        start.extend(vec![
            "cr", "thr", "str", "br", "iv", "est", "ost", "ing", "kr", "in", "on", "tr", "tw",
            "wh", "eld", "ar", "or", "ear", "irr", "mi", "en", "ed", "et", "ow", "fr", "shr", "wr",
            "gr", "pr",
        ]);
        let mut middle = cons.clone();
        middle.extend(vec!["tt"]);
        let vowel = vec!["o", "e", "a", "i", "u", "au", "ee", "ow", "ay", "ey", "oe"];
        let end = vec![
            "et", "ige", "age", "ist", "en", "on", "og", "end", "ind", "ock", "een", "edge", "ist",
            "ed", "est", "eed", "ast", "olt", "ey", "ean", "ead", "onk", "ink", "eon", "er", "ow",
            "ot", "in", "on", "id", "ir", "or", "in", "ig", "en",
        ];

        let mut name = String::new();

        name += start.choose(self.rng).unwrap();
        for _ in 0..self.approx_syllables.saturating_sub(2) {
            name += vowel.choose(self.rng).unwrap();
            name += middle.choose(self.rng).unwrap();
        }
        name += end.choose(self.rng).unwrap();

        name.chars()
            .enumerate()
            .map(|(i, c)| if i == 0 { c.to_ascii_uppercase() } else { c })
            .collect()
    }

    // biome naming
    // generic
    pub fn generate_biome(self) -> String {
        let cons = vec![
            "b", "d", "f", "g", "h", "k", "l", "m", "n", "s", "t", "w", "br", "dr", "gr", "gh",
            "kh", "kr", "st", "str", "th", "tr", "ar", "ark", "adr", "ath", "an", "el", "elb",
            "eldr", "estr", "ostr", "ond", "ondr", "ul", "uld", "eld", "eldr",
        ];
        let start = cons.clone();
        let mid = vec![
            "br", "d", "dr", "dn", "dm", "fr", "g", "gr", "gl", "k", "kr", "l", "ll", "m", "mm",
            "n", "nn", "nd", "st", "th", "rw", "nw", "thr", "lk", "nk", "ng", "rd", "rk", "nr",
            "nth", "rth", "kn", "rl", "gg", "lg", "str", "nb", "lb", "ld", "rm", "sd", "sb",
        ];
        let mut middle = mid.clone();
        middle.extend(vec!["tt"]);
        let vowel = vec!["o", "e", "a", "u", "ae"];
        let end = vec![
            "ul", "um", "un", "uth", "und", "ur", "an", "a", "ar", "a", "amar", "amur", "ath",
            "or", "on", "oth", "omor", "omur", "omar", "ador", "odor", "en", "end", "eth", "amon",
            "edur", "aden", "oden", "alas", "elas", "alath", "aloth", "eloth", "eres", "ond",
            "ondor", "undor", "andor", "od", "ed", "amad", "ud", "amud", "ulud", "alud", "allen",
            "alad", "and", "an", "as", "es",
        ];

        let mut name = String::new();

        name += start.choose(self.rng).unwrap();
        for _ in 0..self.approx_syllables_long.saturating_sub(2) {
            name += vowel.choose(self.rng).unwrap();
            name += middle.choose(self.rng).unwrap();
        }
        name += end.choose(self.rng).unwrap();

        name.chars()
            .enumerate()
            .map(|(i, c)| if i == 0 { c.to_ascii_uppercase() } else { c })
            .collect()
    }

    // biome specific options: engl / custom
    fn generate_engl_from_parts(&mut self, start: &[&str], end: &[&str]) -> String {
        let mut name = String::new();
        name += start.choose(self.rng).unwrap();
        name += end.choose(self.rng).unwrap();

        name.chars()
            .enumerate()
            .map(|(i, c)| if i == 0 { c.to_ascii_uppercase() } else { c })
            .collect()
    }

    fn generate_custom_from_parts(
        &mut self,
        start: &[&str],
        middle: &[&str],
        vowel: &[&str],
        end: &[&str],
    ) -> String {
        let mut name = String::new();

        name += start.choose(self.rng).unwrap();
        for _ in 0..self.approx_syllables.saturating_sub(2) {
            name += vowel.choose(self.rng).unwrap();
            name += middle.choose(self.rng).unwrap();
        }
        name += end.choose(self.rng).unwrap();

        name.chars()
            .enumerate()
            .map(|(i, c)| if i == 0 { c.to_ascii_uppercase() } else { c })
            .collect()
    }

    pub fn generate_grassland_engl(&mut self) -> String {
        let start = [
            "green", "heather", "flower", "blue", "yellow", "vast", "moon", "star", "sun",
            "bright", "fair", "calm", "mistral", "whisper", "clover", "sooth", "spring", "morrow",
            "dim", "dusk", "dawn", "night", "shimmer", "silver", "gold", "amber", "quiet", "still",
            "gleam", "wild", "corm", "mint", "petal", "feather", "silent", "bronze", "bistre",
            "thistle", "bristle", "dew", "bramble", "sorrel", "broad",
        ];
        let end = [
            "brook", "well", "flight", "more", "heart", "song", "barb", "wort", "hoof", "foot",
            "herd", "path", "wind", "breeze", "light", "bloom", "rest", "balm", "reach", "flow",
            "graze", "trail", "fall", "shrub", "bush", "gale", "run", "stem", "glare", "gaze",
            "rove", "brew", "rise", "glow", "wish", "will", "walk", "wander", "wake", "sky",
            "burrow", "cross", "roam",
        ];
        self.generate_engl_from_parts(&start, &end)
    }

    pub fn generate_ocean_engl(&mut self) -> String {
        let start = [
            "dark", "murk", "shimmer", "glimmer", "spirit", "moon", "dead", "bane", "salt",
            "brine", "glitter", "thunder", "pale", "wail", "star", "dim", "azure", "sky", "rage",
            "ripple", "gloom", "sun", "gleam", "void", "storm", "death", "silver", "dread",
            "sorrow", "grim", "spirit",
        ];
        let end = [
            "wave", "shell", "wind", "breeze", "gale", "crash", "grip", "fog", "mirror", "whirl",
            "whorl", "stream", "current", "drift", "light", "shine", "haze", "tear", "maw", "fang",
            "surf", "tide", "rush", "surge", "glow", "drop", "drag", "swell", "scale", "song",
            "shroud", "grasp", "mist",
        ];
        self.generate_engl_from_parts(&start, &end)
    }

    pub fn generate_mountain_engl(&mut self) -> String {
        let start = [
            "white", "frost", "ever", "pale", "hoar", "cold", "chill", "shiver", "bitter",
            "glimmer", "winter", "algor", "grey", "dragon", "thunder", "pallid", "death", "spirit",
            "crystal", "shimmer", "silver", "grizzle", "quiet", "still", "high", "blind", "silent",
            "lost", "somber", "silent", "sharp", "somber", "sharp", "rime", "ice", "spear",
            "hammer", "sword", "war", "sky", "heaven", "dread",
        ];
        let end = [
            "cloud", "veil", "shroud", "fang", "horn", "bite", "more", "howl", "heart", "song",
            "scream", "draft", "path", "wind", "breeze", "wail", "crag", "bellow", "breach",
            "rift", "chasm", "climb", "fall", "rise", "grasp", "grip", "gale", "summit", "shard",
            "pierce", "crush", "shard", "clash", "wish", "will", "wake", "storm", "blade", "hold",
            "reach", "maw", "lock", "gust", "stone", "fury", "rage", "jaw", "gorge", "clove",
        ];
        self.generate_engl_from_parts(&start, &end)
    }

    pub fn generate_snowland_engl(&mut self) -> String {
        let start = [
            "white", "frost", "ever", "pale", "hoar", "cold", "chill", "shiver", "bitter",
            "glimmer", "winter", "algor", "grey", "ghost", "pearl", "pallid", "spectre", "spirit",
            "crystal", "shimmer", "silver", "grizzle", "quiet", "still", "wild", "blind", "silent",
            "lost", "somber", "sleet", "silent", "sharp", "somber", "sleet", "sharp", "rime",
            "ice",
        ];
        let end = [
            "wood", "veil", "shroud", "fang", "horn", "bite", "more", "howl", "heart", "song",
            "scream", "draft", "path", "wind", "breeze", "wail", "growl", "bellow", "pine", "roam",
            "rest", "shade", "fall", "fir", "grasp", "grip", "gale", "hunt", "run", "shadow",
            "hill", "shard", "blood", "wish", "will", "walk", "wander", "wake", "storm", "blade",
            "hold", "reach", "maw",
        ];
        self.generate_engl_from_parts(&start, &end)
    }

    pub fn generate_desert_engl(&mut self) -> String {
        let start = [
            "dry", "sun", "bright", "death", "dread", "sizzle", "scourge", "dearth", "dust",
            "gold", "amber", "gleam", "dead", "swelter", "arid", "balm", "parch", "demon",
            "scorch", "sear", "gold", "simmer", "doom", "torrid", "drake", "devil", "spirit",
            "soul", "singe", "heat", "torch", "bone", "broil", "dragon", "shrivel", "sorrow",
        ];
        let end = [
            "swarm", "claw", "curse", "quill", "spell", "bite", "thorn", "bane", "grasp", "storm",
            "flare", "wind", "breeze", "howl", "reach", "flow", "sting", "sting", "fang", "blast",
            "veil", "scale", "glare", "gaze", "skull", "prickle", "reach", "needle", "void",
            "rise", "maw", "raze", "mirage", "glow", "wander", "roam", "breath", "shard", "scar",
            "lurk", "cross", "stone", "flash", "dance", "burn", "blaze", "fury", "wound",
            "blister", "mark", "waste", "blood", "rock", "grip", "blight",
        ];
        self.generate_engl_from_parts(&start, &end)
    }

    pub fn generate_swamp_engl(&mut self) -> String {
        let start = [
            "green", "moss", "deep", "moon", "haze", "drizzle", "mizzle", "rain", "bracken",
            "hollow", "mist", "fog", "haze", "tear", "mud", "lost", "spirit", "venom", "glum",
            "murk", "dim", "dusk", "night", "quiet", "dark", "brume", "rot", "drench", "sullen",
            "still", "gleam", "wild", "blind", "gnarl", "silent", "dead", "rust", "soul", "gloom",
            "bramble", "briar", "thorn", "earth", "wither", "tangle", "twist", "fear", "slither",
        ];
        let end = [
            "root", "bark", "vine", "brook", "well", "veil", "mantle", "drop", "fang", "beck",
            "bough", "vale", "whisper", "shroud", "cloak", "muck", "claw", "wane", "call", "glen",
            "rest", "shade", "fall", "grasp", "grip", "water", "glare", "swarm", "creek", "crawl",
            "shadow", "glow", "wood", "wander", "river", "pool", "sting", "shriek", "barb", "silt",
            "clay", "pond", "gorse", "burr", "blight", "wisp", "scale", "rill",
        ];
        self.generate_engl_from_parts(&start, &end)
    }

    pub fn generate_jungle_engl(&mut self) -> String {
        let start = [
            "green", "ever", "briar", "thorn", "deep", "sun", "glare", "tangle", "strangle",
            "twist", "dim", "dark", "shade", "savage", "lost", "mist", "shadow", "red", "scarlet",
            "fog", "fern", "quiet", "gleam", "wild", "blind", "swift", "gnarl", "venom", "spirit",
            "dread", "wonder", "drizzle", "coil", "snarl", "skein", "snare", "murk", "moss",
            "slither",
        ];
        let end = [
            "root", "bark", "vine", "leaf", "more", "bole", "heart", "song", "climb", "talon",
            "bough", "breeze", "light", "branch", "bloom", "vale", "swarm", "prowl", "fang",
            "horn", "fall", "bush", "grasp", "grip", "crawl", "run", "sting", "hunt", "pride",
            "veil", "shroud", "rise", "glow", "will", "wander", "wake", "blossom", "barb", "scale",
            "claw", "catch", "plume", "dance", "maw", "roar", "howl", "shriek", "reach", "drench",
            "flower", "snag",
        ];
        self.generate_engl_from_parts(&start, &end)
    }

    pub fn generate_forest_engl(&mut self) -> String {
        let start = [
            "green", "moss", "ever", "briar", "thorn", "oak", "deep", "moon", "star", "sun",
            "bright", "glare", "fair", "calm", "mistral", "whisper", "clover", "hollow", "spring",
            "morrow", "dim", "dusk", "dawn", "night", "shimmer", "silver", "gold", "fern", "quiet",
            "still", "gleam", "wild", "blind", "swift", "gnarl", "flutter", "silent", "honey",
            "bramble", "rose", "aspen",
        ];
        let end = [
            "root", "bark", "log", "brook", "well", "shire", "leaf", "more", "bole", "heart",
            "song", "dew", "bough", "path", "wind", "breeze", "light", "branch", "bloom", "vale",
            "glen", "rest", "shade", "fall", "sward", "shrub", "bush", "grasp", "grip", "gale",
            "crawl", "run", "shadow", "rise", "glow", "wish", "will", "walk", "wander", "wake",
            "eye", "blossom", "sprout", "barb",
        ];
        self.generate_engl_from_parts(&start, &end)
    }

    pub fn generate_forest_custom(&mut self) -> String {
        let cons = vec![
            "d", "f", "h", "l", "m", "n", "r", "s", "t", "w", "v", "z", "st", "th", "thr", "str",
            "vr", "wr", "an", "and", "ald", "ath", "as", "all", "amm", "an", "az", "en", "end",
            "eld", "eldr", "es", "ell", "aldr", "astr", "estr", "on", "ond",
        ];
        let start = cons.clone();
        let middle = vec![
            "th", "d", "dr", "sl", "sm", "rw", "rv", "l", "ll", "m", "mm", "n", "nn", "nd", "ndr",
            "thr", "nth", "ln", "lm", "lz", "nz", "nr", "str", "sth", "ld", "rm", "sd", "rn", "ss",
            "sn", "lw", "lv", "th", "rw", "nw", "r", "rth", "rz",
        ];
        let vowel = vec!["o", "e", "a", "u", "ae"];
        let end = vec![
            "anir", "enir", "elen", "alen", "ezen", "azen", "yn", "andir", "endir", "ond", "alar",
            "ales", "eles", "arain", "alain", "anuin", "athuin", "aruin", "yr", "ethuin", "eruin",
            "athul", "edor", "edes", "edus", "inyr", "eden", "azar", "atiel", "etiel", "etien",
            "eties", "aties", "aduil", "eduil", "eduin", "aduin", "assil", "iryl", "aryl", "anor",
            "anes", "en", "es", "ys",
        ];
        self.generate_custom_from_parts(&start, &middle, &vowel, &end)
    }

    pub fn generate_savannah_engl(&mut self) -> String {
        let start = [
            "red", "pride", "dry", "copper", "vast", "war", "star", "sun", "bright", "blood",
            "dawn", "shimmer", "gold", "amber", "gleam", "wild", "rust", "fire", "earth", "spirit",
            "bronze", "broad", "scorch", "sear", "gold", "prowl", "marl", "rage", "ghost", "soul",
            "specter", "scald", "singe", "heat", "torch", "rite", "bone",
        ];
        let end = [
            "dust", "grass", "swarm", "more", "heart", "song", "claw", "fang", "hoof", "foot",
            "herd", "path", "wind", "breeze", "howl", "sway", "rest", "reach", "flow", "graze",
            "trail", "sting", "fall", "growl", "mane", "bush", "gale", "run", "field", "glare",
            "gaze", "wallow", "brew", "rise", "glow", "wade", "wander", "wake", "sky", "roam",
            "breath", "shard", "scar", "lurk", "hill", "blaze", "fury", "hunt",
        ];
        self.generate_engl_from_parts(&start, &end)
    }

    pub fn generate_taiga_engl(&mut self) -> String {
        let start = [
            "green", "blue", "ever", "pale", "needle", "cold", "moon", "star", "chill", "shiver",
            "bitter", "fair", "calm", "mistral", "whisper", "grey", "hollow", "morrow", "dim",
            "dusk", "dawn", "night", "shimmer", "silver", "iron", "quill", "grizzle", "quiet",
            "still", "wild", "blind", "silent", "somber", "sleet", "silent", "sharp", "somber",
            "sleet", "sharp", "rime", "drizzle", "resin",
        ];
        let end = [
            "root", "bark", "log", "brook", "well", "shire", "more", "bole", "heart", "song",
            "dew", "bough", "path", "wind", "breeze", "light", "branch", "bloom", "pine", "spruce",
            "rest", "shade", "fall", "fir", "grasp", "grip", "gale", "hunt", "run", "shadow",
            "hill", "shadow", "larch", "rise", "bite", "wish", "will", "walk", "wander", "wake",
            "stone", "howl", "moss",
        ];
        self.generate_engl_from_parts(&start, &end)
    }

    // themes & sites
    fn generate_theme_from_parts(
        &mut self,
        start: &[&str],
        middle: &[&str],
        vowel: &[&str],
        end: &[&str],
    ) -> String {
        let mut name = String::new();

        name += start.choose(self.rng).unwrap();
        for _ in 0..self.approx_syllables.saturating_sub(2) {
            name += vowel.choose(self.rng).unwrap();
            name += middle.choose(self.rng).unwrap();
        }
        name += end.choose(self.rng).unwrap();

        name.chars()
            .enumerate()
            .map(|(i, c)| if i == 0 { c.to_ascii_uppercase() } else { c })
            .collect()
    }

    // towns
    pub fn generate_town(mut self) -> String {
        let start = vec![
            "b", "f", "g", "d", "h", "c", "m", "l", "n", "p", "r", "s", "t", "w", "v", "z", "qu",
            "br", "bl", "ch", "chr", "dr", "dw", "fr", "fl", "gr", "gw", "pr", "pl", "st", "sl",
            "str", "sn", "sp", "spr", "sw", "tr", "wr", "ab", "abr", "al", "ald", "as", "ast",
            "amm", "ach", "adr", "en", "end", "eld", "end", "es", "amr", "on", "ond", "ochr",
            "orn", "ost", "ord",
        ];
        let middle = vec![
            "b", "g", "d", "c", "m", "l", "n", "p", "r", "s", "t", "v", "br", "ch", "chr", "dr",
            "gr", "pr", "st", "sl", "sn", "sp", "sw", "tr", "mm", "n", "nn", "nd", "ln", "lm",
            "nr", "r", "rz", "lz", "ld", "rm", "sd", "rn", "ss", "sm", "rv", "lv",
        ];
        let vowel = vec!["e", "a", "i", "o"];
        let end = vec![
            "on", "aton", "enton", "eau", "eaux", "erg", "enberg", "enburg", "oc", "ic", "oric",
            "aric", "eric", "aven", "elen", "andam", "endam", "estin", "ostin", "alic", "elic",
            "alon", "elon", "arac", "erac", "eden", "elen", "owan", "owen", "iel", "ien", "alton",
            "ing", "iling", "aling", "olm", "ulm", "ough", "ilm", "alm", "elm", "eim", "oria",
            "aria", "eria", "elis", "alis", "amor", "emor", "en", "edge", "ard", "erd", "ord",
            "emberg", "emburg", "org", "alton", "agrad", "ograd", "inton",
        ];
        self.generate_theme_from_parts(&start, &middle, &vowel, &end)
    }

    // greek-latin inspired location names for danari
    pub fn generate_danari(mut self) -> String {
        let start = vec![
            "d", "ph", "r", "st", "t", "s", "p", "th", "br", "tr", "m", "k", "cr", "phr", "dr",
            "pl", "ch", "l", "ap", "akr", "ak", "ar", "ath", "asp", "al", "aph", "aphr", "oph",
            "or", "ok", "on", "od", "oth", "om", "ep", "er", "em", "eph", "eth", "yps", "yph",
            "ach", "amph", "yp", "ik", "is", "iph", "ith", "pr", "as", "asph", "ps", "b", "n", "z",
            "x", "kr", "kt", "cht", "chr", "thr", "dr", "pr", "pl", "h", "in", "g", "sph",
        ];
        let middle = vec![
            "kr", "tr", "str", "rk", "st", "n", "r", "ph", "phr", "ch", "x", "d", "l", "kt", "pr",
            "ll", "pp", "ss", "th", "mm", "s", "t", "g", "mn", "rg", "b", "p", "ps", "kl", "dr",
            "mp", "sp", "cht", "lph",
        ];
        let vowel = vec![
            "o", "e", "a", "i", "y", "eo", "ae", "ea", "oi", "io", "ia", "aeo",
        ];
        let end = vec![
            "ilia", "ilios", "os", "oros", "ophos", "elos", "ethos", "athos", "yros", "a", "ares",
            "aros", "olis", "ophis", "era", "ara", "ora", "antis", "entis", "ead", "aeon", "on",
            "eon", "oron", "ena", "is", "as", "eris", "eras", "inis", "aros", "in", "orea", "isis",
            "okles", "akles", "ilion", "anos", "akos", "akon", "enon", "es", "aros", "ikron",
            "orea", "area", "ilon", "ilos", "aelos", "yron", "iron", "adalos", "anon", "ix", "ox",
            "alea", "atheas", "eas", "eos", "yros", "ophon",
        ];
        self.generate_theme_from_parts(&start, &middle, &vowel, &end)
    }

    // primitive-orcish inspired location names for gnarling fortification
    pub fn generate_gnarling(mut self) -> String {
        let start = vec!["gn", "kr", "k", "r", "t", "kn", "tr", "kt", "gr"];
        let middle = vec![
            "t", "tt", "k", "kk", "r", "r", "rl", "lm", "km", "tm", "kn", "kr", "tr", "nk", "gn",
            "kl", "kt", "lt", "arln", "ln", "k't", "k'n", "k'm", "g'm", "l'k", "t'n", "r'k",
            "n'kr", "k R", "t K", "rl Gn", "rl K", "k Gn", "t M", "t N", "r K", "r N", "k M",
            "k T", "rl T", "t Kn", "r Kn",
        ];
        let vowel = vec!["e", "a", "i", "o"];
        let end = vec![
            "arak", "orok", "arok", "orak", "attak", "akarl", "okarl", "atok", "anak", "etak",
            "orek", "arek", "atik", "arik", "etik", "arlak", "arlek", "otek", "almek", "arlnok",
            "arlnak", "okorl", "eknok", "ottok", "erlek", "akkat", "okkar", "attor", "ittor",
            "aktor", "okomor", "imor", "inork", "inor", "amakkor", "ikkor", "amarl", "omarl",
            "ikkarl", "okkarl", "emekk", "akatak", "okatak",
        ];
        self.generate_theme_from_parts(&start, &middle, &vowel, &end)
    }
}
