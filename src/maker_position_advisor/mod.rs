mod profit_counter;
mod mock_trade_tracker;

use mock_trade_tracker::MockTradeTracker;
use profit_counter::ProfitCounter;


pub struct MakerPositionAdvisor{
    profit_counter: ProfitCounter,
    mock_trade_tracker: MockTradeTracker
}

impl MakerPositionAdvisor {
    pub fn new() {

    }
}