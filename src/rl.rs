use indicatif::ProgressIterator;
use reinforcement_learning::blackjack::{NextStateResult, State, next_state};
use std::cmp::Ordering;

use rand::Rng;

#[derive(Clone, Copy, Default)]
pub struct RunningAverage {
    num: i64,
    den: i64,
}

impl RunningAverage {
    pub fn observe(&mut self, value: i64) {
        self.num += value;
        self.den += 1;
    }
}

impl PartialEq for RunningAverage {
    fn eq(&self, other: &Self) -> bool {
        self.den != 0 && other.den != 0 && self.num * other.den == other.num * self.den
    }
}

impl PartialOrd for RunningAverage {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.den == 0 || other.den == 0 {
            None
        } else {
            Some((self.num * other.den).cmp(&(other.num * self.den)))
        }
    }
}

fn explore_start<R: Rng>(rng: &mut R) -> (State, bool) {
    (
        State {
            sum: rng.random_range(12..21),
            dealer_card: rng.random_range(0..10),
            usable_ace: rng.random_bool(0.5),
        },
        rng.random_bool(0.5),
    )
}

fn main() {
    let mut state_action_value_table: [[RunningAverage; 2]; 180] = [[Default::default(); 2]; 180];
    let mut rng = rand::rng();

    // Perform reinforcement learning with exploring starts
    for _ in (0..10000000).progress() {
        let (start_state, start_hit) = explore_start(&mut rng);
        let mut hit = start_hit;
        let mut state = start_state;
        let result = loop {
            match next_state(state, hit, &mut rng) {
                NextStateResult::Some(new_state) => {
                    state = new_state;
                    let action_values = &state_action_value_table[usize::from(&state)];
                    hit = action_values[1] > action_values[0];
                }
                NextStateResult::End(new_result) => {
                    break new_result;
                }
            }
        };
        state_action_value_table[usize::from(&start_state)][start_hit as usize]
            .observe(result as i64);
    }

    // Evaluate greedy policy
    let mut policy = [false; 180];
    for state_index in 0..180 {
        let action_values = &state_action_value_table[state_index];
        policy[state_index] = action_values[1] > action_values[0];
    }

    // Print policy
    for usable_ace in [false, true] {
        println!("Usable ace: {}", usable_ace);
        println!("   2 3 4 5 6 7 8 9 X A");
        for sum in 12..21 {
            print!("{}", sum);
            for dealer_card in 2..12 {
                let state_index = u8::from(&State {
                    usable_ace,
                    sum,
                    dealer_card: dealer_card % 10,
                }) as usize;
                print!(" {}", if policy[state_index] { "H" } else { "S" });
            }
            println!();
        }
    }
}
