use crate::Client;
use common::cmd::*;

trait TabComplete {
    fn complete(&self, part: &str, client: &Client) -> Vec<String>;
}

impl TabComplete for ArgumentSpec {
    fn complete(&self, part: &str, client: &Client) -> Vec<String> {
        match self {
            ArgumentSpec::PlayerName(_) => complete_player(part, client),
            ArgumentSpec::SiteName(_) => complete_site(part, client),
            ArgumentSpec::Float(_, x, _) => {
                if part.is_empty() {
                    vec![format!("{:.1}", x)]
                } else {
                    vec![]
                }
            },
            ArgumentSpec::Integer(_, x, _) => {
                if part.is_empty() {
                    vec![format!("{}", x)]
                } else {
                    vec![]
                }
            },
            ArgumentSpec::Any(_, _) => vec![],
            ArgumentSpec::Command(_) => complete_command(part),
            ArgumentSpec::Message(_) => complete_player(part, client),
            ArgumentSpec::SubCommand => complete_command(part),
            ArgumentSpec::Enum(_, strings, _) => strings
                .iter()
                .filter(|string| string.starts_with(part))
                .map(|c| c.to_string())
                .collect(),
            ArgumentSpec::Boolean(_, part, _) => vec!["true", "false"]
                .iter()
                .filter(|string| string.starts_with(part))
                .map(|c| c.to_string())
                .collect(),
        }
    }
}

fn complete_player(part: &str, client: &Client) -> Vec<String> {
    client
        .player_list
        .values()
        .map(|player_info| &player_info.player_alias)
        .filter(|alias| alias.starts_with(part))
        .cloned()
        .collect()
}

fn complete_site(mut part: &str, client: &Client) -> Vec<String> {
    if let Some(p) = part.strip_prefix('"') {
        part = p;
    }
    client
        .sites
        .values()
        .filter_map(|site| match site.site.kind {
            common_net::msg::world_msg::SiteKind::Cave => None,
            _ => site.site.name.as_ref(),
        })
        .filter(|name| name.starts_with(part))
        .map(|name| {
            if name.contains(' ') {
                format!("\"{}\"", name)
            } else {
                name.clone()
            }
        })
        .collect()
}

fn complete_command(part: &str) -> Vec<String> {
    let part = part.strip_prefix('/').unwrap_or(part);

    ChatCommand::iter_with_keywords()
        .map(|(kwd, _)| kwd)
        .filter(|kwd| kwd.starts_with(part))
        .map(|kwd| format!("/{}", kwd))
        .collect()
}

// Get the byte index of the nth word. Used in completing "/sudo p /subcmd"
fn nth_word(line: &str, n: usize) -> Option<usize> {
    let mut is_space = false;
    let mut j = 0;
    for (i, c) in line.char_indices() {
        match (is_space, c.is_whitespace()) {
            (true, true) => {},
            (true, false) => {
                is_space = false;
                j += 1;
            },
            (false, true) => {
                is_space = true;
            },
            (false, false) => {},
        }
        if j == n {
            return Some(i);
        }
    }
    None
}

pub fn complete(line: &str, client: &Client) -> Vec<String> {
    let word = if line.chars().last().map_or(true, char::is_whitespace) {
        ""
    } else {
        line.split_whitespace().last().unwrap_or("")
    };
    if line.starts_with('/') {
        let mut iter = line.split_whitespace();
        let cmd = iter.next().unwrap();
        let i = iter.count() + if word.is_empty() { 1 } else { 0 };
        if i == 0 {
            // Completing chat command name
            complete_command(word)
        } else if let Ok(cmd) = cmd.parse::<ChatCommand>() {
            if let Some(arg) = cmd.data().args.get(i - 1) {
                // Complete ith argument
                arg.complete(word, client)
            } else {
                // Complete past the last argument
                match cmd.data().args.last() {
                    Some(ArgumentSpec::SubCommand) => {
                        if let Some(index) = nth_word(line, cmd.data().args.len()) {
                            complete(&line[index..], client)
                        } else {
                            vec![]
                        }
                    },
                    Some(ArgumentSpec::Message(_)) => complete_player(word, client),
                    _ => vec![], // End of command. Nothing to complete
                }
            }
        } else {
            // Completing for unknown chat command
            complete_player(word, client)
        }
    } else {
        // Not completing a command
        complete_player(word, client)
    }
}
