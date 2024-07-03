use hftbacktest::depth::{
    ApplySnapshot, HashMapMarketDepth, MarketDepth, INVALID_MAX, INVALID_MIN,
};
use hftbacktest::{
    backtest::reader::Data,
    types::{Event, BUY, SELL},
};
use nalgebra::{
    convert, matrix, ComplexField, Const, DMatrix, DVector, Dyn, Matrix, Matrix1, VecStorage,
    Vector, ViewStorage, U1,
};
use statrs::distribution::{Continuous, LogNormal};
use statrs::euclid::Modulus;
use std::f64::consts::{E, PI};

use super::side_depth::SideDepth;
use super::{EvaluateMatrix, LobMatrix, MAX_DEPTH};

type MatrixView<'a, T> = Matrix<T, Dyn, Dyn, ViewStorage<'a, T, Dyn, Dyn, Const<1>, Dyn>>;
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
    fair_price_tick: f64,
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
            fair_price_tick: 0.0,
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
        self.fair_price_tick = fair_price as f64 / self.tick_size as f64;
        self.hit_prob_coef1 = hit_prob_coef1;
        self.hit_prob_coef2 = hit_prob_coef2;
        self.hit_distance = hit_distance;
        self.hang_distance_profit = hang_distance_profit;
        self.hit_std = hit_std;
    }

    #[inline(always)]
    fn log_normal_pdf(m: f64, std: f64, x: f64) -> f64 {
        (-0.5 * ((x.ln() - m) / std).powi(2)).exp() / ((2.0 * PI).powf(0.5) * std * x)
    }

    fn eval_side(&self, obi: f64, side_lob: LobMatrix) -> DMatrix<f64> {
        let prob_of_best_ask_hit = sigmoid(self.hit_prob_coef1 * obi + self.hit_prob_coef2);
        // let log_normal_pdf = LogNormal::new(self.hit_distance, self.hit_std).unwrap();
        let f32_matrix: DMatrix<f64> = convert(side_lob);
        let ticks = f32_matrix.view((0, 0), (270, 1));
        let qtys = f32_matrix.view((0, 1), (270, 1));
        let dists = ticks.add_scalar(-self.fair_price_tick).abs();
        let pdfs = dists.map(|xi| Self::log_normal_pdf(self.hit_distance, self.hit_std, xi));
        let weights = pdfs.component_mul(&qtys);
        let weight_sum = weights.sum();

        let mut acc: f64 = 0.0;
        let mut result: Vec<f64> = vec![];
        for i in weights.iter().rev() {
            acc += i;
            result.push(acc);
        }
        result.reverse();
        let result_matrix = DVector::from_vec(result);
        convert(result_matrix / weight_sum * prob_of_best_ask_hit)
    }

    pub fn eval(&self) -> (DMatrix<f64>, DMatrix<f64>) {
        let ask_qty = self.ask_depth.best_qty_lot() as f64;
        let bid_qty = self.bid_depth.best_qty_lot() as f64;
        let ask_obi = (bid_qty - ask_qty) / (bid_qty + ask_qty);
        let bid_obi = (ask_qty - bid_qty) / (bid_qty + ask_qty);

        let ask_result = self.eval_side(ask_obi, self.ask_depth.depth.clone());
        let bid_result = self.eval_side(bid_obi, self.bid_depth.depth.clone());
        return (bid_result, ask_result);
    }
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

#[cfg(test)]
mod tests {
    use hftbacktest::depth::{HashMapMarketDepth, MarketDepth};

    use crate::depth::analyticmarketdepth::AnalyticMarketDepth;

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

    #[test]
    fn eval() {
        let mut my_depth = AnalyticMarketDepth::new(0.01, 0.01);

        my_depth.update_ask_depth(100.2, 100.2, 0);
        my_depth.update_ask_depth(100.4, 100.4, 0);
        my_depth.update_ask_depth(100.3, 100.3, 0);
        my_depth.update_ask_depth(100.1, 100.1, 0);

        my_depth.update_bid_depth(90.3, 100.2, 0);
        my_depth.update_bid_depth(90.4, 100.4, 0);
        my_depth.update_bid_depth(90.5, 100.3, 0);
        my_depth.update_bid_depth(90.1, 100.1, 0);
        // my_depth.feed_parameter(95.1, hit_prob_coef1, hit_prob_coef2, hit_distance, hit_std, hang_distance_profit);
        let res = my_depth.eval();
        println!("{:?}", res);
    }
}
