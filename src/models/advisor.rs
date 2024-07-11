use std::{cell::RefCell, collections::HashMap, f64::EPSILON, rc::Rc};

// 指令信息: oid, is_sell, px, qty, action(1-撤单, 2-挂单, 3-对价take) 
pub type OrderInfo = (i64, bool, f64, f64, usize);  
// 账户信息: 当前偏离, 最大偏离限制, 报单速度反馈(1-偏高 0-适中 -1-偏低), 活跃挂单<order_id, orderinfos>
pub type AccountInfo<'a> = (f64, f64, i32, &'a HashMap<i64, OrderInfo>);
pub struct Advisor{
    maker_fee:f32,
    taker_fee:f32,
    tick_size:f32,
    minimum_trade_qty: f64,
    fair_price:f32,
    fair_std:f32,   // fair_price的每秒波动率
    depth_px: (Vec<f32>, Vec<f32>),    // ask depth price, bid depth price, 以盘口由近到远排序
    lob_eval_func: Rc<RefCell<dyn Fn(f64, bool) -> (f64, f64, f64)>>,
    min_retry_level: i64,   // 当前挂单超出最优前m挂单位时,撤单重挂. m根据报单频率限制调整. min_retry_level是m的最小值.
    max_retry_level: i64,
    order_num_limit: i32   //单边挂单数量限制(默认5)
}

impl Advisor{
    fn new(maker_fee:f32, taker_fee:f32, tick_size:f32, minimum_trade_qty:f64, lob_eval_func: Rc<RefCell<dyn Fn(f64, bool) -> (f64, f64, f64)>>)->Self{
        // 初始化一些市场基础信息
        Advisor{
            maker_fee:maker_fee,
            taker_fee:taker_fee,
            tick_size:tick_size,
            minimum_trade_qty: minimum_trade_qty,
            fair_price: f32::NAN,
            fair_std: f32::NAN,
            depth_px:(Vec::<f32>::new(), Vec::<f32>::new()),
            lob_eval_func: lob_eval_func,
            min_retry_level: 5,
            max_retry_level: 20,
            order_num_limit: 5
        }
    }

    pub fn feed_market_info(&mut self, fair:f32, fair_vol:f32, depth_px:(Vec<f32>, Vec<f32>)){
        self.fair_price = fair;
        self.fair_std = fair_vol;
        self.depth_px = depth_px;
    }

    /**
     * 计算随机收益效用. 
     *  采用对数效用函数,即utility = log(1+return).
     *  用泰勒展开到2阶,简化积分运算. 
     * mean:收益的均值;
     * std: 收益的标准差
    */
    fn utility_of_normal_rtn(mean:f64, std:f64)->f64{
        (mean+1.0).ln() - std.powf(2.0)/((1.0+mean).powf(2.0)*2.0)
    }

    fn close_decision_utility(&self, px:&f64, deal_time:&f64, profit:&f64, is_maker:&bool)->(f64,f64,bool,f64){
        let util = Self::utility_of_normal_rtn(1.0+profit, deal_time.sqrt()* self.fair_std as f64);
        (*px, util, *is_maker, *deal_time)
    } 

    fn open_decision_avg_utility(&self, px:&f64, deal_time:&f64, deal_profit:&f64, is_maker:&bool, utility_of_close:&f64, close_time:&f64)->(f64,f64,bool,f64){
        let util = Self::utility_of_normal_rtn(1.0+deal_profit, 0.0) + utility_of_close;
        let cycle_time = deal_time+close_time;
        (*px, util/cycle_time, *is_maker, cycle_time)
    }

    /**
     * 根据账户状态、策略状态和行情信息返回最优决策.
     * 账户状态由trader模块维护; 策略状态由trader模块保管,但由本方法维护; 行情状态由advisor维护.
     * 
     * strategy_status: 存储策略状态.由advisor编辑和修改, 与账户无关.
     *  参数: m:当平/开仓挂单位置超出前m个最优档时, 撤掉重挂.
     * account_info: 账户状态，信息由trade模块给出
     *    deviation:f64 账户当前库存(在现货时，表示当前库存与额定库存的偏移)
     *    max_deviation:f64 账户允许的最大偏离
     *    freq_adj: i32 当前报单频率反馈(1过高/0适当/-1过低)
     *    my_asks:Vec<OrderInfo> 账户当前买方挂单
     *    my_bids:vec<OrderInfo> 账户当前卖方挂单
    */
    pub fn give_advice(&self, account_info:AccountInfo, strategy_status:&mut HashMap<&'static str, i64>)->Vec<OrderInfo>{
        let (deviation, max_deviation, freq_adj, listing_orders) = account_info;

        let mut ask_orders = Vec::<OrderInfo>::with_capacity(self.order_num_limit as usize);
        let mut bid_orders = ask_orders.clone();

        for (&oid, &order) in listing_orders.iter(){
            let is_sell = order.1;
            if is_sell{
                ask_orders.push(order);
            }else{
                bid_orders.push(order);
            }
        }

        let (lob_ask_px, lob_bid_px) = &self.depth_px;

        let mut m = *strategy_status.entry("m").or_insert(10);   //最优档位滑倒m档后则撤单
        let open_flag:bool;  

        match (freq_adj, m) {
            (_f, _m) if (_f<0)&(_m>self.min_retry_level)=> {
                m = ((_m as f64 / 1.2) as i64).max(self.min_retry_level);
                open_flag = true;
            },
            (_f, _m) if (_f>0)&(_m<self.max_retry_level)=> {
                m = ((_m as f64 * 1.2) as i64).min(self.max_retry_level);
                open_flag = true;
            },
            (_f, _m) if (_f>0)&(_m==self.max_retry_level)=> {
                open_flag = false;
            },
            (_, _)=>{
                open_flag = true;
            }
        }

        strategy_status.insert("m", m);

        ask_orders.sort_by_key(|oif|(oif.2/(self.tick_size as f64)) as i64);
        bid_orders.sort_by_key(|oif|(-oif.2/(self.tick_size as f64)) as i64);

        let eval_f = self.lob_eval_func.borrow();
        // (价格, 成交耗时, 成交收益, 是否maker)
        let mut base_sell_income = Vec::<(f64, f64, f64, bool)>::with_capacity(lob_ask_px.len());
        let mut base_buy_income = Vec::<(f64, f64, f64, bool)>::with_capacity(lob_ask_px.len());
        
        let taker_sell_profit = (lob_bid_px[0] - self.fair_price) as f64 / self.fair_price as f64 - self.taker_fee as f64;
        base_sell_income.push((lob_bid_px[0] as f64, 0.0, taker_sell_profit, false));
        for i in 1..lob_ask_px.len(){
            let ask_px = (lob_ask_px[i] - self.tick_size) as f64;
            let (_, deal_exp_time, deal_profit_bps) = eval_f(ask_px, true);
            let profit = deal_profit_bps/10000.0-self.maker_fee as f64;
            base_sell_income.push((ask_px, deal_exp_time, profit, true));
        }
        let taker_buy_profit = (self.fair_price - lob_ask_px[0]) as f64 / self.fair_price as f64 - self.taker_fee as f64;
        base_buy_income.push((lob_ask_px[0] as f64, 0.0, taker_buy_profit, false));
        for i in 1..lob_bid_px.len(){
            let bid_px = (lob_bid_px[i] + self.tick_size) as f64;
            let (_, deal_exp_time, deal_profit_bps) = eval_f(bid_px, true);
            let profit = deal_profit_bps/10000.0-self.maker_fee as f64;
            base_sell_income.push((bid_px, deal_exp_time, profit, true));
        }

        // 开始计算各种平仓选择的效用,并将效用由高到低排序 (价格, 效用, 是否maker, 成交耗时)
        let mut sell_close_utility: Vec<(f64, f64, bool, f64)> = base_sell_income.iter().map(|(px, util, is_maker, deal_time)|self.close_decision_utility(px, util, is_maker, deal_time)).collect();
        let mut buy_close_utility: Vec<(f64, f64, bool, f64)> = base_buy_income.iter().map(|(px, util, is_maker, deal_time)|self.close_decision_utility(px, util, is_maker, deal_time)).collect();
        sell_close_utility.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        buy_close_utility.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let (_, best_sell_close_util, _, best_sell_close_time) = sell_close_utility.get(0).unwrap();
        let (_, best_buy_close_util, _, best_buy_close_time) = buy_close_utility.get(0).unwrap();
        
        // 开始计算各种开仓选择的效用, 并将效用由高到低排序(价格, [开+平]单位时平均效用, 是否maker, [开+平]耗时)
        let mut sell_open_avg_utility:Vec<(f64, f64, bool, f64)> = base_sell_income.iter().map(|(px, deal_time, deal_profit, is_maker)|self.open_decision_avg_utility(px, deal_time, deal_profit, is_maker, best_buy_close_util, best_buy_close_time)).collect();
        let mut buy_open_avg_utility:Vec<(f64, f64, bool, f64)> = base_buy_income.iter().map(|(px, deal_time, deal_profit, is_maker)|self.open_decision_avg_utility(px, deal_time, deal_profit, is_maker, best_sell_close_util, best_sell_close_time)).collect();

        sell_open_avg_utility.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        buy_open_avg_utility.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut action_to_take:Vec<OrderInfo> = vec![];  //所有待执行的交易操作

        // 生成ask方调整建议
        let mut sell_order_num:i32 = 0;
        let mut sell_close_sum_qty: f64 = 0.0;
        let mut sell_open_sum_qty:f64 = 0.0;
        // 判定ask方的挂单是否该撤
        for (oid, is_ask, px, qty, action) in ask_orders{
            let sell_close_remain = deviation - sell_close_sum_qty;
            if  sell_close_remain > EPSILON{
                // 当前是平仓单
                let (_, deal_exp_time, deal_profit_bps) = eval_f(px, is_ask);
                let (_, util, _, _) = self.close_decision_utility(&px, &deal_exp_time, &deal_profit_bps, &true);
                let prefer_idx = match sell_close_utility.binary_search_by(|probe| probe.1.partial_cmp(&(-&util)).unwrap()){
                    Ok(idx)=> {idx},
                    Err(idx)=>{idx}
                };
                if (prefer_idx as i64 > m) | (sell_order_num >= self.order_num_limit){
                    // 撤单
                    action_to_take.push((oid, true, px, qty, 1));
                }else{
                    // 保留, 但要登记该挂单的存在: 占用挂单数, 已平仓额, 以及可能的已开仓额
                    sell_order_num += 1;
                    sell_close_sum_qty += qty.min(sell_close_remain);
                    sell_open_sum_qty += qty - qty.min(sell_close_remain);
                }
            }else{
                // 当前是开仓单
                let (_, deal_exp_time, deal_profit_bps) = eval_f(px, is_ask);
                let (_, avg_util, _, _) = self.open_decision_avg_utility(&px, &deal_exp_time, &deal_profit_bps, &true, best_buy_close_util, best_buy_close_time);
                let prefer_idx = match sell_open_avg_utility.binary_search_by(|probe| probe.1.partial_cmp(&(-&avg_util)).unwrap()){
                    Ok(idx)=> {idx},
                    Err(idx)=>{idx}
                };
                if (prefer_idx as i64 > m)|(avg_util < EPSILON)|(sell_order_num >= self.order_num_limit)|(deviation+max_deviation-sell_open_sum_qty<EPSILON)|!open_flag{
                    // 撤单
                    action_to_take.push((oid, true, px, qty, 1));
                }else{
                    // 保留, 但要登记该开仓挂单的存在: 占用挂单数, 已挂开额, 已开仓额
                    sell_order_num += 1;
                    sell_open_sum_qty += qty;
                }
            }
        }
        let mut suggested_sell_qty_limit = if self.order_num_limit-sell_order_num - 1>0 {
            (deviation + max_deviation - (sell_open_sum_qty+sell_close_sum_qty))/(self.order_num_limit-sell_order_num - 1) as f64
        }else{
            f64::INFINITY
        };
        suggested_sell_qty_limit = suggested_sell_qty_limit.max(self.minimum_trade_qty);
        // ask挂平仓单
        let mut level_cursor:usize = 0;
        while (sell_order_num<self.order_num_limit) & (deviation - sell_close_sum_qty>EPSILON){
            let close_advise = sell_close_utility.get(level_cursor).unwrap();
            let is_maker = close_advise.2;
            let px = close_advise.0;
            let qty = (deviation - sell_close_sum_qty).min(suggested_sell_qty_limit);
            if is_maker{
                action_to_take.push((0, true, px, qty, 2))
            }else{
                action_to_take.push((0, true, px, qty, 3))
            }
            sell_close_sum_qty += qty;
            sell_order_num += 1;
            level_cursor += 1;
        }
        // ask挂开仓单
        level_cursor = 0;
        while (sell_order_num<self.order_num_limit) & (max_deviation-sell_open_sum_qty>EPSILON)&open_flag{
            let open_advise = sell_open_avg_utility.get(level_cursor).unwrap();
            let is_maker = open_advise.2;
            let px = open_advise.0;
            let qty = (max_deviation-sell_open_sum_qty).min(suggested_sell_qty_limit);
            if is_maker{
                action_to_take.push((0, true, px, qty, 2))
            }else{
                action_to_take.push((0, true, px, qty, 3))
            }
            sell_open_sum_qty += qty;
            sell_order_num += 1;
            level_cursor += 1;
        }

        // 生成bid方调整建议
        let mut buy_order_num:i32 = 0;
        let mut buy_close_sum_qty: f64 = 0.0;
        let mut buy_open_sum_qty:f64 = 0.0;
        // 判定bid方的挂单是否该撤
        for (oid, is_ask, px, qty, action) in bid_orders{
            let buy_close_remain = -deviation - buy_close_sum_qty;
            if  buy_close_remain > EPSILON{
                // 当前是平仓单
                let (_, deal_exp_time, deal_profit_bps) = eval_f(px, is_ask);
                let (_, util, _, _) = self.close_decision_utility(&px, &deal_exp_time, &deal_profit_bps, &true);
                let prefer_idx = match buy_close_utility.binary_search_by(|probe| probe.1.partial_cmp(&(-&util)).unwrap()){
                    Ok(idx)=> {idx},
                    Err(idx)=>{idx}
                };
                if (prefer_idx as i64 > m) | (buy_order_num >= self.order_num_limit){
                    // 撤单
                    action_to_take.push((oid, true, px, qty, 1));
                }else{
                    // 保留, 但要登记该挂单的存在: 占用挂单数, 已平仓额, 以及可能的已开仓额
                    buy_order_num += 1;
                    buy_close_sum_qty += qty.min(buy_close_remain);
                    buy_open_sum_qty += qty - qty.min(buy_close_remain);
                }
            }else{
                // 当前是开仓单
                let (_, deal_exp_time, deal_profit_bps) = eval_f(px, is_ask);
                let (_, avg_util, _, _) = self.open_decision_avg_utility(&px, &deal_exp_time, &deal_profit_bps, &true, best_sell_close_util, best_sell_close_time);
                let prefer_idx = match buy_open_avg_utility.binary_search_by(|probe| probe.1.partial_cmp(&(-&avg_util)).unwrap()){
                    Ok(idx)=> {idx},
                    Err(idx)=>{idx}
                };
                if (prefer_idx as i64 > m)|(avg_util < EPSILON)|(buy_order_num >= self.order_num_limit)|(deviation+max_deviation-buy_open_sum_qty<EPSILON)|!open_flag{
                    // 撤单
                    action_to_take.push((oid, true, px, qty, 1));
                }else{
                    // 保留, 但要登记该开仓挂单的存在: 占用挂单数, 已挂开额, 已开仓额
                    buy_order_num += 1;
                    buy_open_sum_qty += qty;
                }
            }
        }
        let mut suggested_buy_qty_limit = if self.order_num_limit-buy_order_num - 1>0 {
            (-deviation + max_deviation - (buy_open_sum_qty+buy_close_sum_qty))/(self.order_num_limit-buy_order_num - 1) as f64
        }else{
            f64::INFINITY
        };
        suggested_buy_qty_limit = suggested_buy_qty_limit.max(self.minimum_trade_qty);
        // bid挂平仓单
        let mut level_cursor:usize = 0;
        while (buy_order_num<self.order_num_limit) & (-deviation - buy_close_sum_qty>EPSILON){
            let close_advise = buy_close_utility.get(level_cursor).unwrap();
            let is_maker = close_advise.2;
            let px = close_advise.0;
            let qty = (-deviation - buy_close_sum_qty).min(suggested_buy_qty_limit);
            if is_maker{
                action_to_take.push((0, true, px, qty, 2))
            }else{
                action_to_take.push((0, true, px, qty, 3))
            }
            buy_close_sum_qty += qty;
            buy_order_num += 1;
            level_cursor += 1;
        }
        // bid挂开仓单
        level_cursor = 0;
        while (buy_order_num<self.order_num_limit) & (max_deviation-buy_open_sum_qty>EPSILON)&open_flag{
            let open_advise = buy_open_avg_utility.get(level_cursor).unwrap();
            let is_maker = open_advise.2;
            let px = open_advise.0;
            let qty = (max_deviation-buy_open_sum_qty).min(suggested_buy_qty_limit);
            if is_maker{
                action_to_take.push((0, true, px, qty, 2))
            }else{
                action_to_take.push((0, true, px, qty, 3))
            }
            buy_open_sum_qty += qty;
            buy_order_num += 1;
            level_cursor += 1;
        }
        action_to_take
    }
}