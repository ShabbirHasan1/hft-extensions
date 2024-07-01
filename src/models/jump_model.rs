use std::f64::NAN;

use hftbacktest::types::Event;
use crate::tools::push_ring_buffer;
use ringbuf::{traits::Consumer, HeapRb};
use linfa_logistic::{FittedLogisticRegression, LogisticRegression};
use linfa_linear::{FittedLinearRegression, LinearRegression};
use linfa::prelude::*;
use ndarray::{array, Array1, Array2};


pub struct JumpModel{
    elapse: i64,  // 观察jump的时间区间(单位:ns)
    observe_bbo:bool,
    last_bbo_timestamp:i64,
    last_bbo:(f64, f64, f64),   // best_bid_price, best_ask_price, obi
    last_max_px:f64,
    last_min_px:f64,
    jump_events:HeapRb<(f64,f64)>,  // jump_dist_bps, obi
    history_sample_count: usize,
    sample_len:usize,
    recalc_len:usize,
    break_model: Option<FittedLogisticRegression<f64, i32>>,  //coef, intercept
    dist_model: Option<FittedLinearRegression<f64>>,
    dist_model_err: f64,
    newest_obi:f64
}

impl JumpModel{
    pub fn new(elapse: i64, sample_len:usize, recalc_len:usize)->Self{
        JumpModel{
            elapse: elapse,
            observe_bbo:true,
            last_bbo:(f64::NAN,f64::NAN,f64::NAN),
            last_bbo_timestamp:0,
            last_max_px:NAN,
            last_min_px:NAN,
            jump_events:HeapRb::<(f64, f64)>::new(sample_len),
            history_sample_count:0,
            sample_len:sample_len,
            recalc_len: recalc_len,
            break_model:None,
            dist_model:None,
            dist_model_err: NAN,
            newest_obi:NAN
        }
    }

    /**
     * 更新模型数据:
     * 0、交替记录bbo和击穿距离
     * 1、更新击穿距离时，将结果计入jump栈
     * 2、样本达到重估间隔时, 重新估算击穿概率\击穿深度\击穿std
    */
    pub fn update_data(&mut self, best_bid:f64, best_ask:f64, obi:f64, bbo_timestamp:i64, trades:&Vec<Event>){
        self.newest_obi = obi;
        if self.observe_bbo{
            self.last_bbo_timestamp = bbo_timestamp;
            self.last_bbo = (best_bid, best_ask, obi);
            self.observe_bbo = false;
            (self.last_max_px, self.last_min_px) = (NAN, NAN);
        }else{
            for trade in trades{
                // 如果发现时间已超过观察限值:存储这次jump观察结果
                if trade.exch_ts > self.last_bbo_timestamp + self.elapse{
                    let (last_bid, last_ask, obi) = self.last_bbo;
                    // 记录bid侧击穿
                    let sell_jump_bps = if last_bid > self.last_min_px{
                        (last_bid-self.last_min_px)*10000.0/last_bid
                    }else{0.0};
                    push_ring_buffer(&mut self.jump_events, (sell_jump_bps, -obi));
                    // 记录ask侧击穿
                    let ask_jump_bps = if last_ask < self.last_max_px{
                        (self.last_max_px-last_ask)*10000.0/last_ask
                    }else{0.0};
                    push_ring_buffer(&mut self.jump_events, (ask_jump_bps, obi));
                    self.history_sample_count += 2;
                    // 触发击穿概率/击穿距离的重新评估
                    if (self.history_sample_count > self.sample_len)&((self.history_sample_count/2)%(self.recalc_len/2)==0){
                        self.evaluate();
                    }
                    // 继续观测bbo
                    self.observe_bbo = true;
                }else if trade.exch_ts>self.last_bbo_timestamp{
                    let px_f64 = trade.px as f64;
                    if !(self.last_max_px > px_f64){
                        self.last_max_px = px_f64
                    }
                    if !(self.last_min_px < px_f64){
                        self.last_min_px = px_f64
                    }
                }
            }
        }
    }

    /**
     *  重估jump的统计性质，更新：
     * 1、log_jump的平均值
     * 2、log_jump的std
     * 3、一档挂单击穿概率的模型参数
    */
    fn evaluate(&mut self){
        let mut x = Array2::from_elem((self.sample_len,1), f64::NAN);
        let mut y = Array1::from_elem(self.sample_len, 0_i32);

        let mut x_vec = Vec::<f64>::new();
        let mut ln_dist_vec = Vec::<f64>::new();

        let mut dists = Array1::from_elem(self.sample_len, f64::NAN);
        for (idx, (j_dist,obi)) in self.jump_events.iter().enumerate(){
            x[[idx,0]] = *obi;
            dists[idx] = *j_dist;
            if *j_dist > 0.0{
                y[idx] = 1;
                x_vec.push(*obi);
                ln_dist_vec.push(j_dist.ln());
            }else{
                y[idx] = 0;
            }
        }
        let dataset_brk = Dataset::new(x.clone(), y);
        let break_model:FittedLogisticRegression<f64, i32> = LogisticRegression::default()
        .max_iterations(100)
        .fit(&dataset_brk)
        .expect("Failed to fit jump prob model");
        self.break_model = Some(break_model);
        
        let linreg_x = Array2::<f64>::from_shape_vec((x_vec.len(), 1), x_vec).unwrap();
        let linreg_dist = Array1::<f64>::from_shape_vec(ln_dist_vec.len(), ln_dist_vec).unwrap();
        let dataset_dist = Dataset::new(linreg_x.clone(), linreg_dist.clone());
        let dist_model:FittedLinearRegression<f64> = LinearRegression::new()
        .fit(&dataset_dist)
        .expect("Failed to fit jump dist model");
        self.dist_model = Some(dist_model);
        let predictions = self.dist_model.as_ref().unwrap().predict(&linreg_x);
        let residuals = &linreg_dist - &predictions;
        // 计算模型标准误
        self.dist_model_err = residuals.std(0.0);
    }

    /**
     * best_ask, best_bid在下个elapse观测中被击穿的概率
    */
    pub fn bbo_hitback_prob(&self)->(f64,f64){
        let model = self.break_model.as_ref().unwrap();
        let ask_prob = model.predict_probabilities(&array![[self.newest_obi]])[0];
        let bid_prob = model.predict_probabilities(&array![[-self.newest_obi]])[0];
        (ask_prob, bid_prob)
    }

    /**
     * 对数jump距离的:向上mean、向下mean,以及std
    */
    pub fn jump_statistics(&self)->(f64,f64,f64){
        let model = self.dist_model.as_ref().unwrap();
        let up_dist  = model.predict(&array![[self.newest_obi]])[0];
        let down_dist = model.predict(&array![[-self.newest_obi]])[0];
        (up_dist, down_dist, self.dist_model_err)
    }

}