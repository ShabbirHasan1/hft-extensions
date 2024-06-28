use hftbacktest::prelude::*;
use std::{cell::{Ref, RefCell}, f64::NAN, fmt::Debug, rc::Rc};
use ringbuf::{traits::{Consumer, Observer, Producer}, HeapRb};
use nalgebra::{DMatrix, DVector};
const INVALID_MAX:f32 = f32::MAX;
const INVALID_MIN:f32 = f32::MIN;

fn push_ring_buffer<T:Debug>(rb: &mut HeapRb<T>, value: T)->Option<T>{
    let mut ele: Option<T> = None;
    if rb.is_full() {
        ele = Some(rb.try_pop().unwrap());
    }
    rb.try_push(value).unwrap();
    ele
}

type Vec270 = DVector<f64>;

pub struct DistanceProfitResult(Rc<RefCell<DMatrix<f64>>>);

impl DistanceProfitResult{

    pub fn view(&self)->Ref<'_, DMatrix<f64>>{
        self.0.borrow()
    }
}

pub struct DistanceModel 
where{
    elapse: i64,      // elapse duration, in nanoseconds
    account_ticks: i64,    // 每次elapse都虚拟挂单一次, 总共统计多少次的挂单盈利
    _recalc_ticks: i64,  // 间隔多少个tick重新评估一次挂单收益
    _current_ticknum: i64,  // 当前经历的tick数
    /*为计算挂单是否能开仓所设变量*/
    _offset_round: i64,    // 假设latency是5ms, calc调用的elapse duration是3ms, 那么挂单在下1个(=round(5/3))interval不能成交
    _offset_timespan: i64,  // 同上假设，挂单在下下个interval的头2ms(=5%3)内不能成交.
    /*后半截价格 */
    last_after_max_px: f32,
    last_after_min_px: f32,

    /*临时挂单栈: ringbuffer, 记录未确定哪些能成交的挂单价格*/
    _temp_bid: HeapRb<Vec270>,
    _temp_ask: HeapRb<Vec270>,
    /*成交矩阵: 记录成交的挂单的价格.未成交挂单价格为nan*/
    _bid_deal: HeapRb<Vec270>,
    _ask_deal: HeapRb<Vec270>,
    /*历史fair price*/
    _fair_price: HeapRb<f64>,
    /*结算挂单利润时，用多久以后的fair price */
    _fair_dist: i64,
    /*交易费用*/
    _maker_fee: f64,
    /*挂单距离*/
    distance: DVector<f64>,
    /*预期挂单收益:270*3的矩阵,3列分别为:距离, avg_profit, sum_profit in bps*/
    expected_profit: Rc<RefCell<DMatrix<f64>>>
}

impl DistanceModel
{   
    pub fn new(elapse: i64, latency: i64, account_ticks: i64, fair_dist: i64, maker_fee: f64, recalc_ticks:i64) -> Self {
        /*elapse:观测间隔(ns)
          latency: 挂单延时(ns)
          account_ticks: 观测tick数量
          fair_dist: 计算挂单利润时，利用多久以后的fair当作结算利润. 这里是纳秒数
          recalc_ticks: 刷新评估结果的间隔
         */
        let offset_timespan:i64 = latency % elapse;
        let offset_round:i64 = latency / elapse;
        let mut temp_bid = HeapRb::<Vec270>::new((offset_round+2) as usize);
        let mut temp_ask = HeapRb::<Vec270>::new((offset_round+2) as usize);
        let bid_deal = HeapRb::<Vec270>::new(account_ticks as usize);
        let ask_deal = HeapRb::<Vec270>::new(account_ticks as usize);
        let fair_price = HeapRb::<f64>::new(account_ticks as usize);
        while temp_bid.vacant_len()>0{
            let _ = temp_bid.try_push(DVector::from_vec(vec![NAN; 270]));
            let _ = temp_ask.try_push(DVector::from_vec(vec![NAN; 270]));
        }

        let mut distance = DVector::<f64>::from_element(270, NAN);
        distance.iter_mut().enumerate().for_each(|(i, d)| {
            *d = match i {
                0..=99 => (i+1) as f64 * 0.00001,
                100..=189 => 0.001 + (i - 99) as f64 * 0.0001,
                190..=269 => 0.01 + (i - 189) as f64 * 0.0005,
                _ => NAN,
            };
        });

        let expected_profit = Rc::new(RefCell::new(DMatrix::<f64>::from_element(270, 3, NAN)));
        expected_profit.borrow_mut().set_column(0, &distance);

        DistanceModel { 
            elapse: elapse, 
            account_ticks: account_ticks, // Add this line
            _offset_round: offset_round, 
            _offset_timespan: offset_timespan, 
            last_after_max_px:INVALID_MIN,
            last_after_min_px:INVALID_MAX,
            _temp_bid: temp_bid, 
            _temp_ask: temp_ask, 
            _bid_deal: bid_deal, 
            _ask_deal: ask_deal,
            _fair_price: fair_price,
            _fair_dist:fair_dist,
            _maker_fee:maker_fee,
            distance:distance,
            expected_profit: expected_profit,
            _recalc_ticks:recalc_ticks,
            _current_ticknum:0
        }     }
    
    pub fn get_report(&self)->DistanceProfitResult{
        DistanceProfitResult(Rc::clone(&self.expected_profit))
    } 

    fn evaluate(&mut self){
        /*计算不同挂单距离下的利润*/
        let fair_tick_dist = self._fair_dist/self.elapse;
        let shift = fair_tick_dist-(self._offset_round+1);

        let mut count=Vec270::from_vec(vec![0.0; 270]);
        let mut profit_sum = Vec270::from_vec(vec![0.0; 270]);
        
        let fair_prices: Vec<_> = self._fair_price.iter().collect();
        let ask_deal_prices: Vec<_> = self._ask_deal.iter().collect();
        let bid_deal_prices: Vec<_> = self._bid_deal.iter().collect();

        for i in 0..(self.account_ticks-shift){
            let fair = *fair_prices[i as usize];
            let ask_deals = ask_deal_prices[(i+shift) as usize];
            let bid_deals = bid_deal_prices[(i+shift) as usize];
            for j in 0..270{
                let ap = ask_deals[j];
                let bp = bid_deals[j];
                if !ap.is_nan(){
                    count[j]+=1.0;
                    profit_sum[j] += ap/fair-1.0-self._maker_fee;
                }
                if !bp.is_nan(){
                    count[j]+=1.0;
                    profit_sum[j] += fair/bp-1.0-self._maker_fee;
                }
            }
        }
        let profit_per_deal = profit_sum.component_div(&count);
        self.expected_profit.borrow_mut().set_column(1, &profit_per_deal);
        self.expected_profit.borrow_mut().set_column(2, &profit_sum);
    }
    
    pub fn feed(&mut self, fair_price:f64, timenow:i64, trades:&Vec<Event>)->(){
        // 更新成交数据
        self.update_data(fair_price, timenow, trades);
        // 在固定时间间隔下更新挂单位置评估
        if (self._current_ticknum>self.account_ticks) & (self._current_ticknum%self._recalc_ticks==0) {
            self.evaluate();
        }
    }

    fn update_data(&mut self, fair_price:f64, timenow:i64, trades:&Vec<Event>) -> () {
        /*更新行情 */
        self._current_ticknum += 1;
        // println!("last trades len {}", trades.len());

        // 获取latency点之前的最大/最小成交价格
        let current_bf_max_px = trades.iter()
        .filter(|trade| trade.local_ts < timenow - self.elapse + self._offset_timespan)
        .map(|trade| trade.px)
        .fold(INVALID_MIN, |a, b| a.max(b));
        let current_bf_min_px = trades.iter()
        .filter(|trade| trade.local_ts < timenow - self.elapse + self._offset_timespan)
        .map(|trade| trade.px)
        .fold(INVALID_MAX, |a, b| a.min(b));

        let best_buy_px = self.last_after_max_px.max(current_bf_max_px);
        let best_sell_px = self.last_after_min_px.min(current_bf_min_px);

        // t时刻的挂单
        let ask_px = fair_price*self.distance.add_scalar(1.0);
        let bid_px = fair_price*(-&self.distance).add_scalar(1.0);

        // 存储当前挂单，然后计算t-(_offset_round+2)时刻的挂单的成交状态
        let mut past_ask_prices = push_ring_buffer(& mut self._temp_ask, ask_px).unwrap();
        past_ask_prices.iter_mut().for_each(|price| {
            if *price >= best_buy_px as f64 {
                *price = NAN;
            }
        });
        
        let mut past_bid_prices = push_ring_buffer(& mut self._temp_bid, bid_px).unwrap();
        past_bid_prices.iter_mut().for_each(|price| {
            if *price <= best_sell_px as f64 {
                *price = NAN;
            }
        });

        // 把成交状态塞进历史成交矩阵. t-(_offset_round+2)时刻的挂单的成交状态.
        push_ring_buffer(& mut self._ask_deal, past_ask_prices);
        push_ring_buffer(& mut self._bid_deal, past_bid_prices);

        // 记录当前fair_price. t时刻
        push_ring_buffer(& mut self._fair_price, fair_price);

        // 获取latency点之后的最大/最小成交价格
        let current_aft_max_px = trades.iter()
        .filter(|trade| trade.local_ts > timenow - self.elapse + self._offset_timespan)
        .map(|trade| trade.px)
        .fold(INVALID_MIN, |a, b| a.max(b));
        let current_aft_min_px = trades.iter()
            .filter(|trade| trade.local_ts > timenow - self.elapse + self._offset_timespan)
            .map(|trade| trade.px)
            .fold(INVALID_MAX, |a, b| a.min(b));

        self.last_after_max_px = current_aft_max_px;
        self.last_after_min_px = current_aft_min_px;

    }

}


#[cfg(test)]
mod tests {

    #[test]
    fn main(){

    }
    

}
