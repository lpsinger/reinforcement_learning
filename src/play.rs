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

use std::cmp::Ordering;

use inquire::Confirm;
use log::info;
use rand::Rng;
use reinforcement_learning::blackjack::{NextStateResult, State, next_state};

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .init();
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
            NextStateResult::Some(new_state) => {
                state = new_state;
            }
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
