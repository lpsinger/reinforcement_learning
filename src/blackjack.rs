use std::{collections::HashMap, hash::Hash};

use rand::Rng;
use reinforcement_learning::rl::train_monte_carlo_exploring_starts;

/// A non-terminal Blackjack state.
#[derive(Clone, Copy, Eq, PartialEq)]
struct State {
    /// Player's current score, in 12..21 for non-terminal states.
    sum: u8,
    /// Dealer's current card, in 0..10.
    dealer_card: u8,
    /// Whether or not the player holds a "usable" ace: an ace that is being counted as 11 points.
    usable_ace: bool,
}

/// Map each non-terminal Blackjack state onto a unique unsigned integer, 0..180.
impl From<&State> for u8 {
    fn from(value: &State) -> Self {
        ((value.sum - 12) * 10 + value.dealer_card) * 2 + value.usable_ace as u8
    }
}

impl Hash for State {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let index: u8 = self.into();
        index.hash(state);
    }
}

/// Draw a random card assuming an infinite deck.
///
/// Cards are represented by unsigned integers, 0..10, as follows:
/// 0 => Ten, Jack, Queen, or King, worth 10 points.
/// 1 => Ace, which may be worth 1 or 11 points.
/// n => Pip card n, worth n points.
fn draw_card<R: Rng>(rng: &mut R) -> u8 {
    rng.random_range(-3i8..10i8).max(0) as u8
}

/// Get the score of a card, treating aces as 1 point.
fn card_value(card: u8) -> u8 {
    if card == 0 { 10 } else { card }
}

/// Play the dealer's hand and return the dealer's final score.
fn play_dealer<R: Rng>(mut dealer_card: u8, rng: &mut R) -> u8 {
    let mut sum = 0;
    let mut usable_ace = false;

    loop {
        sum += card_value(dealer_card);
        if dealer_card == 1 && sum <= 11 {
            sum += 10;
            usable_ace = true;
        } else if sum > 21 && usable_ace {
            sum -= 10;
            usable_ace = false;
        }

        if sum >= 17 {
            return sum;
        }
        dealer_card = draw_card(rng);
    }
}

/// Display a Blackjack policy.
fn display_policy(policy: HashMap<State, bool>) {
    println!("Policy (S for Stick, H for Hit):");
    println!("Usable ace?     No                    Yes");
    println!("Dealer Card     X A 2 3 4 5 6 7 8 9   X A 2 3 4 5 6 7 8 9");
    for sum in 12..21 {
        print!("Player Sum {sum}");
        for usable_ace in [false, true] {
            print!("  ");
            for dealer_card in 0..10 {
                let state = State {
                    sum,
                    dealer_card,
                    usable_ace,
                };
                let symbol = match policy.get(&state) {
                    Some(false) => "S",
                    Some(true) => "H",
                    None => "?",
                };
                print!(" {symbol}");
            }
        }
        println!();
    }
}

fn main() {
    let mut rng = rand::rng();
    let policy: HashMap<State, bool> = train_monte_carlo_exploring_starts(
        10000000,
        |rng| {
            (
                State {
                    sum: rng.random_range(12..21),
                    dealer_card: rng.random_range(0..10),
                    usable_ace: rng.random_bool(0.5),
                },
                rng.random_bool(0.5),
            )
        },
        |(mut state, hit), rng| {
            if hit {
                state.sum += card_value(draw_card(rng));
                if state.sum == 21 {
                    // intentionally left blank
                } else if state.sum < 21 {
                    return (Some(state), 0);
                } else if state.usable_ace {
                    state.sum -= 10;
                    state.usable_ace = false;
                    return (Some(state), 0);
                } else {
                    return (None, -1);
                }
            }

            let dealer_sum = play_dealer(state.dealer_card, rng);
            return (
                None,
                if dealer_sum > 21 {
                    1
                } else {
                    state.sum.cmp(&dealer_sum) as i64
                },
            );
        },
        &mut rng,
    );
    display_policy(policy);
    // The optimal policy should be...
    //
    // Policy (S for Stick, H for Hit):
    // Usable ace?     No                    Yes
    // Dealer Card     X A 2 3 4 5 6 7 8 9   X A 2 3 4 5 6 7 8 9
    // Player Sum 12   H H H H S S S H H H   H H H H H H H H H H
    // Player Sum 13   H H S S S S S H H H   H H H H H H H H H H
    // Player Sum 14   H H S S S S S H H H   H H H H H H H H H H
    // Player Sum 15   H H S S S S S H H H   H H H H H H H H H H
    // Player Sum 16   H H S S S S S H H H   H H H H H H H H H H
    // Player Sum 17   S S S S S S S S S S   H H H H H H H H H H
    // Player Sum 18   S S S S S S S S S S   H H S S S S S S S H
    // Player Sum 19   S S S S S S S S S S   S S S S S S S S S S
    // Player Sum 20   S S S S S S S S S S   S S S S S S S S S S
}
