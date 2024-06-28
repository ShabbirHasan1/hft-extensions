use hftbacktest::depth::{HashMapMarketDepth, L2MarketDepth, MarketDepth};
use std::collections::HashMap;
use hftbacktest::depth::{ApplySnapshot};
use hftbacktest::{
    backtest::reader::Data,
    types::Event,
};
use std::rc::Rc;
use nalgebra::{DMatrix, DVector};
use std::f64::consts::{E, PI};

use super::{EvaluateMatrix, MAX_DEPTH};

/// L2 Market depth implementation based on a Matrix.
/// 维护当前 mid ± 5%, mid±20 hit-dist std, 或者5000个tick的深度的depth. 三者取其小.
/// feed: 实时fair, hit-through-prob, hit-dist std, hit-dist mean, lob update.
/// observe: 观测每个挂单位置的期望成交时间和单位时间挂单利润.
/// _evaluate: 计算每个挂单位置的期望成交时间和单位时间挂单利润. 需要先rotate.
/// eval_interval: 单位时间的长度(ns)，通常为100ms也就是100_000000ns
/// lob_update: 更新bbo位置, 不rotate.
/// resize(): 在hit-dist更新时: 如果距离上次resize超过30秒, rotate并且resize一下Matrix长度.
pub struct AnalyticMarketDepth {
    pub evaluated_ask: EvaluateMatrix, // 对外提供: 单位时间击穿概率(假设在queue第一位)，单位时间挂单收益，期望耗时
    pub evaluated_bid: EvaluateMatrix,
    eval_interval: i64,
    _inner: HashMapMarketDepth,
    _evaluate_timestamp: i64,
    fair_price_tick: i32,
    obi:f64,
    hit_prob_coef1: f64,
    hit_prob_coef2: f64,
    mean_of_log_hit_distance: f64,
    std_of_log_hit_distance: f64,
    hang_distance_profit_bps: Rc<DMatrix<f64>>,
}

impl AnalyticMarketDepth {
    /// Constructs an instance of `HashMapMarketDepth`.
    pub fn new(tick_size: f32, lot_size: f32, eval_interval:i64, hang_distance_profit_bps:Rc<DMatrix<f64>>) -> Self {
        let mat_evaluate = EvaluateMatrix::zeros(MAX_DEPTH as usize, 3);
        Self {
            _evaluate_timestamp: 0,
            evaluated_ask: mat_evaluate.clone(),
            evaluated_bid: mat_evaluate,
            eval_interval:eval_interval,
            _inner: HashMapMarketDepth::new(tick_size, lot_size),
            fair_price_tick: 0,
            obi:0.0,
            hit_prob_coef1: 0.0,
            hit_prob_coef2: 0.0,
            mean_of_log_hit_distance: 0.0,
            std_of_log_hit_distance: 0.0,
            hang_distance_profit_bps:hang_distance_profit_bps
        }
    }

    pub fn feed_parameter(
        &mut self,
        fair_price: f64,
        obi:f64,
        hit_prob_coef1: f64,
        hit_prob_coef2: f64,
        mean_of_log_hit_distance: f64,
        std_of_log_hit_distance: f64,
    ) {
        self.fair_price_tick = (fair_price as f32 / self._inner.tick_size) as i32;
        self.obi = obi;
        self.hit_prob_coef1 = hit_prob_coef1;
        self.hit_prob_coef2 = hit_prob_coef2;
        self.mean_of_log_hit_distance = mean_of_log_hit_distance;
        self.std_of_log_hit_distance = std_of_log_hit_distance;
    }

    #[inline(always)]
    fn log_normal_pdf(m: f64, std: f64, x: f64) -> f64 {
        (-0.5 * ((x.ln() - m) / std).powi(2)).exp() / ((2.0 * PI).powf(0.5) * std * x)
    }


    /**
     * hang_distance_profit: 不同挂单位的命中期望利润
     * 求挂在x这个距离，命中时的期望利润
     * 
     */
    fn interp1d(hang_distance_profit: &DMatrix<f64>, x_new: f64) -> f64 {
        let n = hang_distance_profit.nrows();
        let u = hang_distance_profit.column(0);
        let v = hang_distance_profit.column(1);
        let x = u.as_slice();
        let y = v.as_slice();
        
        // 使用 binary_search_by 查找插入位置
        match x.binary_search_by(|probe| probe.partial_cmp(&x_new).unwrap()) {
            Ok(idx) => y[idx], // 如果找到精确值，直接使用 y[idx]
            Err(idx) => {
                if idx == 0 {
                    y[0] // x_new 在 x 范围左侧
                } else if idx == n {
                    y[y.len() - 1] // x_new 在 x 范围右侧
                } else {
                    // 线性插值计算
                    let x0 = x[idx - 1];
                    let x1 = x[idx];
                    let y0 = y[idx - 1];
                    let y1 = y[idx];
                    y0 + (x_new - x0) * (y1 - y0) / (x1 - x0)
                }
            }
        }
    }

    fn eval_side(&self, depth: HashMap<i32, f32>, ask_flag: i32) -> (DVector<i32>, DMatrix<f64>) {
        let mut side_lob: Vec<(i32, f32)> = depth.into_iter().collect();
        side_lob.sort_by_key(|&(key, _)| ask_flag * key);
        
        // 1. 计算挂单位置的命中概率(假设挂在queue第一位)
        let obi_s = self.obi * ask_flag as f64;
        let prob_of_best_bbo_hit = sigmoid(self.hit_prob_coef1 * obi_s + self.hit_prob_coef2);
        let bbo_price = side_lob.get(0).unwrap().0;
        let bbo_bps_dists = side_lob.iter().map(|&(p_tick, _)| (p_tick - bbo_price) as f32 * 10000.0 * ask_flag as f32 / bbo_price as f32 );
        let pdfs = bbo_bps_dists.map(|xi| Self::log_normal_pdf(self.mean_of_log_hit_distance, self.std_of_log_hit_distance, xi as f64));
        let weights = pdfs.zip(side_lob.iter().map(|&(_, qty)| qty)).map(|(prob_dense, qty)| prob_dense * qty as f64);
        let weight_sum: f64 = weights.clone().skip(1).sum();
        let mut acc: f64 = 0.0;
        let mut hit_prob: Vec<f64> = vec![];
        for i in weights.rev() {
            acc += i;
            hit_prob.push(acc / weight_sum * prob_of_best_bbo_hit);
        }
        hit_prob.reverse();
        hit_prob[0] = 1.0;

        // 2. 计算挂单期望成交时间
        let exp_deal_time_in_second = hit_prob.clone().into_iter().map(|prob| self.eval_interval as f64 / prob / 1_000_000_000.0);
        // 生成结果矩阵
        let mut result_matrix = DMatrix::zeros(side_lob.len(), 2);
        let mut price_vec = DVector::<i32>::zeros(side_lob.len());
        for (i, ((&(p_tick, _), prob), time)) in side_lob.iter()
            .zip(hit_prob.iter())
            .zip(exp_deal_time_in_second)
            .enumerate() {
                price_vec[i] = p_tick; // 价格
                result_matrix[(i, 0)] = *prob; // 命中概率
                result_matrix[(i, 1)] = time; // 期望成交时间
        }
        (price_vec, result_matrix)
    }

    /**
     * 产生一个挂单评估器:
     * 给定任何价位以及方向(是想买还是卖),输出成交概率、期望成交时间、单位时间挂单利润(in bps)
     */
    pub fn get_eval_func(&self) -> Rc<dyn Fn(f64, bool) -> (f64, f64, f64)>{
        let ask_hashmap = self._inner.ask_depth.clone();
        let (ask_price_tick, ask_probs) = self.eval_side(ask_hashmap, 1);
        let bid_hashmap = self._inner.bid_depth.clone();
        let (bid_price_tick, bid_probs) = self.eval_side(bid_hashmap, -1);
        let hang_distance_profit_bps = Rc::clone(&self.hang_distance_profit_bps);
        let fair_price = self.fair_price_tick as f32*self.tick_size();
        let tick_size = self.tick_size();

        let evaluator:Rc<dyn Fn(f64, bool)->(f64,f64,f64)> = Rc::new(move |px:f64, is_ask_quote:bool|{
            let px_tick = (px / tick_size as f64) as i32;
            let reference_idx:usize;
            let prob_book:&DMatrix<f64>;
            let tick_array:&DVector<i32>;
            let fair_dist_bps:f64;
            if is_ask_quote {
                tick_array = &ask_price_tick;
                prob_book = &ask_probs;
                fair_dist_bps = (px-fair_price as f64)*10000.0/fair_price as f64;
                match tick_array.as_slice().binary_search_by(|probe| probe.partial_cmp(&px_tick).unwrap()){
                    Ok(idx)=> {reference_idx = idx + 1}
                    Err(idx)=>{reference_idx = idx}
                }
            }
            else{
                tick_array = &bid_price_tick;
                prob_book = &bid_probs;
                fair_dist_bps = (fair_price as f64-px)/fair_price as f64*10000.0;
                match (-tick_array).as_slice().binary_search_by(|probe| probe.partial_cmp(&(-px_tick)).unwrap()){
                    Ok(idx)=> {reference_idx = idx + 1}
                    Err(idx)=>{reference_idx = idx}
                }
            }

            let hit_profit_bps = Self::interp1d(&hang_distance_profit_bps, fair_dist_bps);
            if reference_idx >= prob_book.shape().0{
                (0.0, f64::INFINITY, 0.0)
            }else{
                let row = prob_book.row(reference_idx);
                (row[0], row[1], row[0]*hit_profit_bps)
            }
            
        });
        return evaluator;
    }
}

impl L2MarketDepth for AnalyticMarketDepth{

    fn update_bid_depth(
        &mut self,
        price: f32,
        qty: f32,
        timestamp: i64,
    ) -> (i32, i32, i32, f32, f32, i64) {
        return self._inner.update_bid_depth(price, qty, timestamp)
    }

    fn update_ask_depth(
        &mut self,
        price: f32,
        qty: f32,
        timestamp: i64,
    ) -> (i32, i32, i32, f32, f32, i64) {
        return self._inner.update_ask_depth(price, qty, timestamp)
    }

    fn clear_depth(&mut self, side: i64, clear_upto_price: f32) {
        self._inner.clear_depth(side, clear_upto_price)
    }

}

impl MarketDepth for AnalyticMarketDepth {

    #[inline(always)]
    fn best_bid(&self) -> f32 {
        self._inner.best_bid()
    }

    #[inline(always)]
    fn best_ask(&self) -> f32 {
        self._inner.best_ask()
    }

    #[inline(always)]
    fn best_bid_tick(&self) -> i32 {
        self._inner.best_bid_tick()
    }

    #[inline(always)]
    fn best_ask_tick(&self) -> i32 {
        self._inner.best_ask_tick()
    }

    #[inline(always)]
    fn tick_size(&self) -> f32 {
        self._inner.tick_size()
    }

    #[inline(always)]
    fn lot_size(&self) -> f32 {
        self._inner.lot_size()
    }

    #[inline(always)]
    fn bid_qty_at_tick(&self, price_tick: i32) -> f32 {
        self._inner.bid_qty_at_tick(price_tick)
    }

    #[inline(always)]
    fn ask_qty_at_tick(&self, price_tick: i32) -> f32 {
        self._inner.ask_qty_at_tick(price_tick)
    }
}

impl ApplySnapshot<Event> for AnalyticMarketDepth {
    fn apply_snapshot(&mut self, data: &Data<Event>) {
        self._inner.apply_snapshot(data);
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

#[cfg(test)]
mod tests {
    use hftbacktest::depth::{HashMapMarketDepth, MarketDepth, L2MarketDepth};
    use nalgebra::{DMatrix, DVector};
    use std::rc::Rc;
    use crate::depth::analyticmarketdepth::AnalyticMarketDepth;

    #[test]
    fn updates() {
        let mut hash_depth = HashMapMarketDepth::new(0.01, 0.01);
        let mut my_depth = AnalyticMarketDepth::new(0.01, 0.01, 100_000_000, Rc::new(DMatrix::<f64>::zeros(270, 2)));
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
        assert!(hash_res == my_res);
        assert!(hash_depth.best_ask() == my_depth.best_ask());
    }

    #[test]
    fn main() {
        let mut profit = DMatrix::<f64>::zeros(270, 2);
        let profit_refer_price = DVector::from_iterator(270, (0..270).map(|x| x as f64));
        let profit_refer_profit = DVector::from_iterator(270, (0..270).map(|x| (x as f64).sqrt()));
        profit.set_column(0,&profit_refer_price);
        profit.set_column(1,&profit_refer_profit);
        let mut my_depth = AnalyticMarketDepth::new(0.01, 0.01, 100_000_000, Rc::new(profit));

        my_depth.update_ask_depth(100.0, 1.0, 0);
        my_depth.update_ask_depth(101.0, 0.1, 0);
        my_depth.update_ask_depth(104.0, 0.4, 0);
        my_depth.update_ask_depth(102.0, 0.2, 0);

        my_depth.update_bid_depth(90.3, 100.2, 0);
        my_depth.update_bid_depth(90.4, 100.4, 0);
        my_depth.update_bid_depth(90.5, 100.3, 0);
        my_depth.update_bid_depth(90.1, 100.1, 0);

        // ->这里认为一档挂单被击穿的概率为0.5.
        let hit_prob_coef1 = 0.0;
        let hit_prob_coef2 = 0.0;
        /*这里,ask1～4的数据：
         *  概率密度分别为: nan, 0.0010925357415163615, 0.001310888336120623, 0.0009728315622231069
         *  加权点概率:   nan, 0.00010925357415163615, 0.0002621776672241246, 0.00038913262488924277
         *  weight_sum(skip1): 0.0007605638662650034
         *  queue1穿透概率: 1, 0.5, 0.42817593696098044, 0.2558185066036632
         */ 
        let mean_of_log_hit_distance = 6.214608;
        let std_of_log_hit_distance = 1.0;
        // 所有距离的命中收益都是100%(10000bps)
        my_depth.feed_parameter(100.0, 0.0, hit_prob_coef1, hit_prob_coef2, mean_of_log_hit_distance, std_of_log_hit_distance);
        /*
         * 挂单利润参考点: 0,0bps; 1,1bps; 2,1.414bps; ..., 100,10bps; 200,10.1bps; 104,10.2bps
         * 这样的话, 挂在ask 1～3档后的成交概率\命中利润\单位时间挂单收益\期望成交时间分别为:
         * 1. 0.5, 0bps, 0, 0.2
         * 2. 0.42817, 10bps, 4.28bps, 0.2335
         * 3. 0.25582, 14.142bps, 3.6178bps, 0.3909
        */
        let evaluator = my_depth.get_eval_func();
        let eval_result_ask3 = evaluator(102.0, true);
        assert!((eval_result_ask3.0-0.255818).abs()<0.0001);  // 挂在ask3末尾的单个interval成交概率
        assert!((eval_result_ask3.1-0.3909).abs()<0.0001);   // 挂在ask3末尾的期望成交时间(秒)
        assert!((eval_result_ask3.2-3.6178).abs()<0.0001);  //  挂在ask3末尾的单个interval期望利润(bps)

    }
}
