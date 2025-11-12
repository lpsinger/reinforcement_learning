use std::{collections::HashMap, hash::Hash};

use indicatif::ProgressIterator;
use num::Rational64;
use rand::Rng;

/// Monte Carlo ES (Exploring Starts), for estimating optimal policy,
/// from Sutton & Barto 2nd Ed. Section 5.3.
pub fn train_monte_carlo_exploring_starts<
    R: Rng,
    State: Copy + Eq + Hash,
    Action: Copy + Default + Eq + Hash,
    ExploreStarts: Fn(&mut R) -> (State, Action),
    NextState: Fn((State, Action), &mut R) -> (Option<State>, i64),
>(
    episodes: usize,
    explore_starts: ExploreStarts,
    next_state: NextState,
    rng: &mut R,
) -> HashMap<State, Action> {
    let mut policy = HashMap::new();
    let mut q: HashMap<State, HashMap<Action, Rational64>> = HashMap::new();
    for _ in (0..episodes).progress() {
        let mut state_action = explore_starts(rng);
        let mut episode: Vec<((State, Action), i64)> = Vec::new();
        loop {
            let (next_state, reward) = next_state(state_action, rng);
            episode.push((state_action, reward));
            if let Some(state) = next_state {
                state_action = (
                    state,
                    match policy.get(&state) {
                        Some(&action) => action,
                        None => Action::default(),
                    },
                );
            } else {
                break;
            }
        }

        let mut returns = 0;
        let mut first_visits: HashMap<(State, Action), i64> = HashMap::new();
        for &(state_action, reward) in episode.iter().rev() {
            returns += reward;
            first_visits.insert(state_action, returns);
        }

        for (&(state, action), &returns) in first_visits.iter() {
            let qq = q.entry(state).or_default();
            qq.entry(action)
                .and_modify(|f| *f = Rational64::new_raw(f.numer() + returns, f.denom() + 1))
                .or_insert_with(|| Rational64::from_integer(returns));

            let (&action, _) = qq.iter().max_by_key(|&(_, value)| value).unwrap();
            policy.insert(state, action);
        }
    }
    return policy;
}
