use hftbacktest::depth::{
    ApplySnapshot, HashMapMarketDepth, MarketDepth, INVALID_MAX, INVALID_MIN,
};
use hftbacktest::{
    backtest::reader::Data,
    types::{Event, BUY, SELL},
};
use nalgebra::{convert, matrix, ComplexField, Const, DMatrix, Dyn, Matrix, VecStorage, U1};
use statrs::distribution::{Continuous, LogNormal};
use statrs::euclid::Modulus;
use std::f64::consts::E;

const MAX_DEPTH: i32 = 5000_i32;
type LobMatrix = DMatrix<i32>;
type EvaluateMatrix = DMatrix<f32>;

pub struct SideDepth {
    pub tick_size: f32,
    pub lot_size: f32,
    pub depth: LobMatrix,
    pub best_index: i32,
    pub tick_ord: i32,
}

impl SideDepth {
    // tick_cmp > 0 升序， 否则降序
    pub fn new(tick_size: f32, lot_size: f32, tick_ord: i32) -> Self {
        let mut init = LobMatrix::zeros(MAX_DEPTH as usize, 2);
        return SideDepth {
            tick_size,
            lot_size,
            depth: init,
            best_index: -1, // -1 表示没有挂单
            tick_ord: tick_ord,
        };
    }

    pub fn update(
        &mut self,
        price: f32,
        qty: f32,
        timestamp: i64,
    ) -> (i32, i32, i32, f32, f32, i64) {
        let price_tick = (price / self.tick_size).round() as i32;
        let head_index = self.best_index;

        // 如果当前挂单为空， 则直接挂到index0
        if head_index == -1 {
            let mut item = self.depth.row_mut(0_usize);
            item[0] = price_tick;
            let qty_lot = (qty / self.lot_size).round() as i32;
            item[1] = qty_lot;
            self.best_index = 0;
            return (
                price_tick,
                if self.tick_ord > 0 {
                    INVALID_MAX
                } else {
                    INVALID_MIN
                },
                self.depth.row(self.best_index as usize)[1],
                0.0,
                qty,
                timestamp,
            );
        }

        let head = self.depth.row(head_index as usize);
        let head_tick = head[0];
        let head_qty_lot = head[1];
        let offset = price_tick - head_tick;

        if offset >= MAX_DEPTH as i32 {
            // 忽略太远的位置
            return (
                price_tick,
                head_tick,
                self.depth.row(self.best_index as usize)[1],
                head_qty_lot as f32 * self.lot_size,
                qty,
                timestamp,
            );
        }

        let index = (head_index + offset).rem_euclid(MAX_DEPTH as i32);
        let mut item = self.depth.row_mut(index as usize);
        item[0] = price_tick;
        let qty_lot = (qty / self.lot_size).round() as i32;
        let prev_qty = item[1] as f32 * self.lot_size;
        item[1] = qty_lot;
        // 如果是0, 表示该位置没有挂单
        if qty_lot == 0 {
            // item[0] = INVALID_MIN
        }
        // 更新首尾指针
        // 价格前进了，
        if offset < 0 {
            self.best_index = index;
        }
        // 盘口撤单或被吃， 价格后退, 找到下一个qty不为0的价位作为best
        if offset == 0 && qty_lot == 0 {
            for i in 1..MAX_DEPTH {
                let index = (self.best_index) + i.rem_euclid(MAX_DEPTH as i32);
                if self.depth.row(index as usize)[1] != 0_i32 {
                    self.best_index = index;
                    break;
                }
            }
            // todo 虽然不太可能， 挂单被清空了怎么办？
        }
        // todo 尾指针需要维护吗？
        (
            price_tick,
            head_tick,
            self.depth.row(self.best_index as usize)[0],
            prev_qty,
            qty,
            timestamp,
        )
    }

    pub fn clear_depth(&mut self, clear_upto_price: f32) {
        // 暂时没有需求
        // let tick_upto = (clear_upto_price / self.tick_size).round() as i32;
    }

    pub fn best_tick(&self) -> i32 {
        self.depth.row(self.best_index as usize)[0]
    }

    pub fn best_price(&self) -> f32 {
        self.best_tick() as f32 * self.tick_size
    }

    pub fn qty_at_tick(&self, tick: i32) -> f32 {
        self.depth.row(self.tick_to_index(tick) as usize)[1] as f32 * self.lot_size
    }

    fn tick_to_index(&self, tick: i32) -> i32 {
        let head_index = self.best_index;
        let head = self.depth.row(head_index as usize);
        let head_tick = head[0];
        let offset = tick - head_tick;
        if offset >= MAX_DEPTH as i32 {
            panic!("too large tick");
        }
        let index = (head_index + offset).rem_euclid(MAX_DEPTH as i32);
        return index;
    }
}

/// L2 Market depth implementation based on a Matrix.
/// 维护当前 mid ± 5%, mid±20 hit-dist std, 或者5000个tick的深度的depth. 三者取其小.
/// feed: 实时fair, hit-through-prob, hit-dist std, hit-dist mean, lob update.
/// observe: 观测每个挂单位置的期望成交时间和单位时间挂单利润.
/// _evaluate: 计算每个挂单位置的期望成交时间和单位时间挂单利润. 需要先rotate.
/// lob_update: 更新bbo位置, 不rotate.
/// resize(): 在hit-dist更新时: 如果距离上次resize超过30秒, rotate并且resize一下Matrix长度.
pub struct AnalyticMarketDepth {
    pub tick_size: f32,
    pub lot_size: f32,
    pub ask_depth: SideDepth,
    pub bid_depth: SideDepth,
    pub evaluated_ask: EvaluateMatrix, // 对外提供: 单位时间击穿概率，单位时间挂单收益，期望耗时
    pub evaluated_bid: EvaluateMatrix,

    _evaluate_timestamp: i64,
    fair_price: f64,
    hit_prob_coef1: f64,
    hit_prob_coef2: f64,
    hit_distance: f64,
    hit_std: f64,
    hang_distance_profit: DMatrix<f64>,
}

impl AnalyticMarketDepth {
    /// Constructs an instance of `HashMapMarketDepth`.
    pub fn new(tick_size: f32, lot_size: f32) -> Self {
        let mat_evaluate = EvaluateMatrix::zeros(MAX_DEPTH as usize, 3);
        Self {
            tick_size,
            lot_size,
            ask_depth: SideDepth::new(tick_size, lot_size, 1),
            bid_depth: SideDepth::new(tick_size, lot_size, -1),
            _evaluate_timestamp: 0,
            evaluated_ask: mat_evaluate.clone(),
            evaluated_bid: mat_evaluate,
            fair_price: 0.0,
            hit_prob_coef1: 0.0,
            hit_prob_coef2: 0.0,
            hit_distance: 0.0,
            hit_std: 0.0,
            hang_distance_profit: DMatrix::<f64>::zeros(270, 2),
        }
    }

    pub fn feed_parameter(
        &mut self,
        fair_price: f64,
        hit_prob_coef1: f64,
        hit_prob_coef2: f64,
        hit_distance: f64,
        hit_std: f64,
        hang_distance_profit: DMatrix<f64>,
    ) {
        self.fair_price = fair_price;
        self.hit_prob_coef1 = hit_prob_coef1;
        self.hit_prob_coef2 = hit_prob_coef2;
        self.hit_distance = hit_distance;
        self.hang_distance_profit = hang_distance_profit;
        self.hit_std = hit_std;
    }

    // fn eval_side(&self, obi: f64, side_lob: LobMatrix) {
    //     let prob_of_best_ask_hit = sigmoid(self.hit_prob_coef1 * obi + self.hit_prob_coef2);
    //     let log_normal_pdf = LogNormal::new(self.hit_distance, self.hit_std).unwrap();
    //     log_normal_pdf.pdf(1.0);
    //     let distances = side_lob.column(0).rows(0, 270);
    // }
}

impl MarketDepth for AnalyticMarketDepth {
    /**
     * 单侧订单薄用一个循环数组表示
     * 从index=0开始生长, 随时记录订单薄首尾的下标
     * 更新步骤：
     * 计算 price_tick,
     * offset_tick_from_head = (price_tick - head_tick)
     * if offset_tick_from_head >= len, 说明价格离盘口太远, 忽略该价位更新
     * index = (head_index + offset_tick_from_head)  % len
     * 更新index处的价格和qty
     *
     * 如果更新的是首尾处， 同时更新首尾指针
     */
    fn update_bid_depth(
        &mut self,
        price: f32,
        qty: f32,
        timestamp: i64,
    ) -> (i32, i32, i32, f32, f32, i64) {
        return self.bid_depth.update(price, qty, timestamp);
    }

    fn update_ask_depth(
        &mut self,
        price: f32,
        qty: f32,
        timestamp: i64,
    ) -> (i32, i32, i32, f32, f32, i64) {
        return self.ask_depth.update(price, qty, timestamp);
    }

    fn clear_depth(&mut self, side: i64, clear_upto_price: f32) {
        if side == BUY {
            self.bid_depth.clear_depth(clear_upto_price);
        } else if side == SELL {
            self.ask_depth.clear_depth(clear_upto_price);
        } else {
            self.bid_depth.clear_depth(clear_upto_price);
            self.ask_depth.clear_depth(clear_upto_price);
        }
    }

    #[inline(always)]
    fn best_bid(&self) -> f32 {
        self.bid_depth.best_price()
    }

    #[inline(always)]
    fn best_ask(&self) -> f32 {
        self.ask_depth.best_price()
    }

    #[inline(always)]
    fn best_bid_tick(&self) -> i32 {
        self.bid_depth.best_tick()
    }

    #[inline(always)]
    fn best_ask_tick(&self) -> i32 {
        self.ask_depth.best_tick()
    }

    #[inline(always)]
    fn tick_size(&self) -> f32 {
        self.tick_size
    }

    #[inline(always)]
    fn lot_size(&self) -> f32 {
        self.lot_size
    }

    #[inline(always)]
    fn bid_qty_at_tick(&self, price_tick: i32) -> f32 {
        self.bid_depth.qty_at_tick(price_tick)
    }

    #[inline(always)]
    fn ask_qty_at_tick(&self, price_tick: i32) -> f32 {
        self.bid_depth.qty_at_tick(price_tick)
    }
}

impl ApplySnapshot for AnalyticMarketDepth {
    fn apply_snapshot(&mut self, data: &Data<Event>) {
        self.bid_depth = SideDepth::new(self.tick_size, self.lot_size, -1);
        self.ask_depth = SideDepth::new(self.tick_size, self.lot_size, 1);
        for row_num in 0..data.len() {
            let price = data[row_num].px;
            let qty = data[row_num].qty;
            if data[row_num].ev & BUY == BUY {
                self.update_bid_depth(price, qty, 0);
            } else if data[row_num].ev & SELL == SELL {
                self.update_ask_depth(price, qty, 0);
            }
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

#[test]
fn updates() {
    let mut hash_depth = HashMapMarketDepth::new(0.01, 0.01);
    let mut my_depth = AnalyticMarketDepth::new(0.01, 0.01);
    let hash_res = hash_depth.update_ask_depth(100.1, 100.1, 0);
    let my_res = my_depth.update_ask_depth(100.1, 100.1, 0);
    assert!(hash_res == my_res);
    assert!(hash_depth.best_ask() == my_depth.best_ask());

    let hash_res = hash_depth.update_ask_depth(100.2, 100.5, 0);
    let my_res = my_depth.update_ask_depth(100.2, 100.5, 0);
    assert!(hash_res == my_res);
    assert!(hash_depth.best_ask() == my_depth.best_ask());

    let hash_res = hash_depth.update_ask_depth(100.0, 100.5, 0);
    let my_res = my_depth.update_ask_depth(100.0, 100.5, 0);
    // println!("{:?} {:?}", hash_res, my_res);
    assert!(hash_res == my_res);
    assert!(hash_depth.best_ask() == my_depth.best_ask());
}
