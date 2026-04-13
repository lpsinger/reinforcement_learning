// Rules of blackjack:
// - Assume an infinite deck so that card counting does not help.
// - Card values:
//      - Aces have the value of 1 or 11, whichever results in the higher score for the player without going bust.
//      - Face cards count as 10.
//      - All other cards have their face value.
// - The objective for each player is to have hand with the highest possible value less than or equal to 21.
// - A hand whose value is over 21 is called a "bust" and is an instant win for the opposite player.
// - At the start of the game, the dealer deals himself one face-up card.
// - The player's turn then commences. The dealer hands cards to the player as long as the player asks for a hit.
// - Once the player's turn has ended, the dealer's turn begins. The dealer hits himself until his score is >= 17, or he has gone bust.

use std::{cmp::Ordering, hash::Hash};

use inquire::Confirm;
use log::info;
use rand::Rng;

/// A non-terminal Blackjack state.
#[derive(Clone, Copy, Eq, PartialEq)]
struct State {
    /// Player's current score, in 12..21 for non-terminal states.
    /// We don't bother representing states with player scores less than 12,
    /// because in those states the player should always hit.
    sum: u8,
    /// Dealer's face-up card, in 0..10.
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
    let card = rng.random_range(-3i8..10i8).max(0) as u8;
    info!("Drew card: {card}");
    return card;
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

pub enum NextStateResult {
    Some(State),
    End(Ordering)
}

fn next_state<R: Rng>(state: State, hit: bool, rng: &mut R) -> NextStateResult {
    let mut state = state;
    if hit {
        state.sum += card_value(draw_card(rng));
        if state.sum == 21 {
            // intentionally left blank
        } else if state.sum < 21 {
            return NextStateResult::Some(state);
        } else if state.usable_ace {
            state.sum -= 10;
            state.usable_ace = false;
            return NextStateResult::Some(state);
        } else {
            info!("Player went bust");
            return NextStateResult::End(Ordering::Less);
        }
    }
    info!("Player score {}. Dealer's turn now", state.sum);

    let dealer_sum = play_dealer(state.dealer_card, rng);
    return NextStateResult::End(if dealer_sum > 21 {
            info!("Dealer went bust");
            Ordering::Greater
        } else {
            info!("Dealer score {dealer_sum}");
            state.sum.cmp(&dealer_sum)
        }
    );
}

fn main() {
    env_logger::Builder::new().filter_level(log::LevelFilter::Info).init();
    let mut rng = rand::rng();
    let mut state = State {
        sum: rng.random_range(12..21),
        dealer_card: rng.random_range(0..10),
        usable_ace: rng.random_bool(0.5),
    };
    let ordering: Ordering;
    loop {
        info!(
            "Your score: {} Usable ace: {} Dealer card: {}",
            state.sum,
            if state.usable_ace { "Y" } else { "N" },
            state.dealer_card
        );
        let hit = Confirm::new("Hit?").prompt().unwrap();
        match next_state(state, hit, &mut rng) {
            NextStateResult::Some(new_state) => {state = new_state;},
            NextStateResult::End(new_ordering) => {
                ordering = new_ordering;
                break;
            }
        }
    }

    match ordering {
        Ordering::Less => info!("Dealer wins"),
        Ordering::Greater => info!("Player wins"),
        Ordering::Equal => info!("Tie"),
    }
}
