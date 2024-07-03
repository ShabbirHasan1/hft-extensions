use std::cmp::{max, min};

use log::info;

pub struct MockTradeTracker {
    epsilon: f64,
    count_num: i32,
    buy_cumulate_qty: f64,
    sell_cumulate_qty: f64,
    buy_cumulate_amt: f64,
    sell_cumulate_amt: f64,

    suggest_bid_price: f64,
    suggest_ask_price: f64,
    best_bid: f64,
    best_ask: f64,

    exist_trade_num: i32,
    maker_fee: f64,
    max_level: i32,
    up_adjust_thresh: f64,
    down_adjust_thresh: f64,

    level: i32,
    begin_real_level: i32,

    hanging_timestamp: u64,

    last_mock_round_begin_time: u64,
    hanging_refresh_interv: u64,
    network_lag: u64,
}
impl MockTradeTracker {
    pub fn new(
        maker_fee: f64,
        up_adjust_thresh: f64,
        down_adjust_thresh: f64,
    ) -> MockTradeTracker {
        let mut tracker = MockTradeTracker::default();
        tracker.epsilon = 0.0000000001;
        tracker.count_num = 100;
        tracker.maker_fee = maker_fee;
        tracker.max_level = 3;
        tracker.up_adjust_thresh = up_adjust_thresh;
        tracker.down_adjust_thresh = down_adjust_thresh;
        tracker.begin_real_level = 2;
        tracker.hanging_refresh_interv = 300;
        tracker.network_lag = 20;
        tracker.level = max(tracker.max_level / 2, 0);
        tracker
    }
    pub fn set_order_price(
        &mut self,
        ask_price: f64,
        bid_price: f64,
        best_ask: f64,
        best_bid: f64,
        lob_timestamp: u64,
    ) {
        if lob_timestamp < self.hanging_timestamp + self.hanging_refresh_interv {
            return;
        }
        self.suggest_bid_price = bid_price;
        self.suggest_ask_price = ask_price;
        self.best_bid = best_bid;
        self.best_ask = best_ask;
        self.hanging_timestamp = lob_timestamp;
    }
    fn position_holding(&self) -> f64 {
        return self.buy_cumulate_qty - self.sell_cumulate_qty;
    }

    fn ask(&self) -> (f64, f64) {
        if self.suggest_ask_price > self.best_ask
            && (self.position_holding() + 1_f64 > self.epsilon)
        {
            return (self.suggest_ask_price, self.position_holding() + 1_f64);
        } else if self.position_holding() > self.epsilon {
            return (self.best_ask, self.position_holding());
        } else {
            return (f64::NAN, 0_f64);
        }
    }

    fn bid(&self) -> (f64, f64) {
        if self.suggest_bid_price < self.best_bid
            && (1_f64 - self.position_holding() > self.epsilon)
        {
            return (self.suggest_bid_price, 1_f64 - self.position_holding());
        } else if -self.position_holding() > self.epsilon {
            return (self.best_bid, -self.position_holding());
        } else {
            return (f64::NAN, 0_f64);
        }
    }

    pub fn can_real(&self) -> bool {
        return self.level >= self.begin_real_level;
    }

    pub fn match_trade(&mut self, trade_ts: u64, trade_price: f64, fairprice: f64) {
        if self.hanging_timestamp + self.network_lag > trade_ts {
            return;
        }
        let (ask_price, ask_qty) = self.ask();
        let (bid_price, bid_qty) = self.bid();
        if (ask_qty > 0_f64) && (trade_price > ask_price) {
            self.exist_trade_num += 1;
            self.sell_cumulate_qty += ask_qty;
            self.sell_cumulate_amt += ask_qty * ask_price;
        }
        if (bid_qty > 0_f64) && (trade_price < bid_price) {
            self.exist_trade_num += 1;
            self.buy_cumulate_qty += bid_qty;
            self.buy_cumulate_amt += bid_qty * bid_price;
        }
        if self.exist_trade_num >= self.count_num {
            let now_ts = trade_ts;
            let average_deal_time_used =
                (now_ts - self.last_mock_round_begin_time) / self.count_num as u64 / 1000_u64;
            self.last_mock_round_begin_time = now_ts;
            let profit_bps = ((fairprice * (self.buy_cumulate_qty - self.sell_cumulate_qty)
                + self.sell_cumulate_amt
                - self.buy_cumulate_amt)
                / (self.sell_cumulate_amt + self.buy_cumulate_amt)
                - self.maker_fee)
                * 10000_f64;

            if profit_bps > self.up_adjust_thresh {
                self.level = min(self.level + 1, self.max_level)
            }
            if !(profit_bps > self.down_adjust_thresh) {
                self.level = max(self.level - 1, 0)
            }
            info!(
                "mock profit: {:?} bps, current level {:?}, average deal time: {:?}",
                ((profit_bps * 10_f64) / 10_f64).round(),
                self.level,
                average_deal_time_used
            );
            self.exist_trade_num = 0;
            self.buy_cumulate_qty = 0.0;
            self.sell_cumulate_qty = 0.0;
            self.buy_cumulate_amt = 0.0;
            self.sell_cumulate_amt = 0.0;
        }
    }
}

impl Default for MockTradeTracker {
    fn default() -> Self {
        Self {
            epsilon: Default::default(),
            count_num: Default::default(),
            buy_cumulate_qty: Default::default(),
            sell_cumulate_qty: Default::default(),
            buy_cumulate_amt: Default::default(),
            sell_cumulate_amt: Default::default(),
            suggest_bid_price: Default::default(),
            suggest_ask_price: Default::default(),
            best_bid: Default::default(),
            best_ask: Default::default(),
            exist_trade_num: Default::default(),
            maker_fee: Default::default(),
            max_level: Default::default(),
            up_adjust_thresh: Default::default(),
            down_adjust_thresh: Default::default(),
            level: Default::default(),
            begin_real_level: Default::default(),
            hanging_timestamp: Default::default(),
            last_mock_round_begin_time: Default::default(),
            hanging_refresh_interv: Default::default(),
            network_lag: Default::default(),
        }
    }
}

#[test]
fn playground() {
    let a = 3 / 2;
    println!("{:?}", a);
}
