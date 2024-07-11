use std::collections::HashMap;

use super::{advisor::{Advisor, OrderInfo, AccountInfo}, profit_recorder::ProfitRecorder};

pub struct MockTrader<'a>{
    advisor:&'a Advisor,
    position:f64,
    profit_recorder: Option<ProfitRecorder>,
    active_orders: HashMap<i64, OrderInfo>,
    revoking: Vec<OrderInfo>,
    current_profit_level: usize,
    max_profit_level: usize,
    strategy_status: HashMap<&'static str, i64>,
    increasing_oid: i64,
    taker_fee:f64,
    maker_fee:f64
}

impl <'a> MockTrader<'a>{
    // 初始化一个新实例.
    // 注入一个用于记录盈利和仓位的ProfitRecorder
    pub fn new(advisor:&'a Advisor, max_profit_level:usize, taker_fee:f64, maker_fee:f64)->Self{
        MockTrader{
            advisor: advisor,
            position:0.0,
            active_orders: HashMap::<i64, OrderInfo>::new(),
            revoking: vec![],
            current_profit_level:1,
            max_profit_level:max_profit_level,
            profit_recorder:None,
            strategy_status: HashMap::<&'static str, i64>::new(),
            increasing_oid:0,
            taker_fee:taker_fee,
            maker_fee:maker_fee
        }
    }


    // best_buyer_price/best_seller_price: 策略loop所流过的一小段时间内，最高的taker_buy_price和最低的taker_sell_price.
    // best_bid/best_ask: 当前bbo.
    // 在每次策略loop时被策略调用.
    // 调用时: 
    //     0、利用最新行情,确定撮合区中的挂单的成交状态,以及更新自身状态（调用settle）
    //     1、用持有的&advisor结合自身field生成一套交易指令.(调用advisor.give_advice)
    //     2、用order_submit方法部署这套交易指令.
    pub fn mock_run(&mut self, best_buyer_price:f64, best_seller_price:f64, best_bid:f64, best_ask:f64, fair:f64){
        self.settle(best_buyer_price, best_seller_price, best_bid, best_ask, fair);
        let account_info:AccountInfo = (self.position, 1.0, -1, &self.active_orders); 
        let decisions:Vec<OrderInfo> = self.advisor.give_advice(account_info, &mut self.strategy_status);
        self.instruction_proceed(decisions);
    }

    // 处理交易指令. 
    // *将挂单、吃单指令置于活跃订单区;
    // *将撤单指令置于待撤区.
    fn instruction_proceed(&mut self, decisions:Vec<OrderInfo>){
        for advice in decisions{
            let (oid, is_sell, px, qty, action) = &advice;
            match action {
                &1=>{
                    self.revoking.push(advice)
                }
                _=>{
                    self.increasing_oid += 1;
                    let order = (self.increasing_oid, *is_sell, *px, *qty, *action);
                    self.active_orders.insert(self.increasing_oid, order);
                }
            }
        }
    }

    // 结算交易指令,
    // 结算动作包括：
    // （撤销 决定吃单请求是否成功; 决定挂单是否成交.）
    // 对于挂单区中的订单: 
    //     如果best_buyer_price/best_seller_price跨越了相应方向的订单，视为该订单成交，此时：
    //         1、修改自身position并推送信息到profit_recorder(调用feed_status方法)
    //         2、删除该挂单.
    // 对于吃单区中的订单：
    //     遍历这里的taker订单，如果best_bid/best_ask满足主卖/主买的价格，则视为taker单成交.此时：
    //         1、修改自身position并推送信息到profit_recorder(调用feed_status方法)
    //     不论是否成交，遍历完后都清空吃单区订单.
    // 最后处理撤单请求.
    fn settle(&mut self, best_buyer_price:f64, best_seller_price:f64, best_bid:f64, best_ask:f64, fair:f64){
        if self.profit_recorder.is_none(){
            self.profit_recorder = Some(ProfitRecorder::new(self.max_profit_level, 1.0*6.0, self.position, fair))
        }
        let pr_ref = self.profit_recorder.as_mut().unwrap();
        // feed新的成交之后, 查看profit_recorder的level是否有改变; 如有, 重设它的bin_size, 并同步自身level.
        // settle过后, 待处理区应当只存在活跃挂单
        let mut order_expired = Vec::<i64>::with_capacity(100);
        for (oid, order) in self.active_orders.iter(){
            match order {
                // 卖挂单成交
                (_, true, px, qty, 2) if best_buyer_price>*px=>{
                    let fee = px*qty*self.maker_fee;
                    self.position -= qty;
                    pr_ref.feed_new_deal(true, *px, *qty, fee);
                    order_expired.push(*oid);
                }
                // 买挂单成交
                (_, false, px, qty, 2) if best_seller_price<*px=>{
                    let fee = px*qty*self.maker_fee;
                    self.position += qty;
                    pr_ref.feed_new_deal(false, *px, *qty, fee);
                    order_expired.push(*oid);
                }
                // 卖吃单处理
                (_, true, px, qty, 3)=>{
                    if best_bid>=*px{
                        let fee = px*qty*self.taker_fee;
                        self.position -= qty;
                        pr_ref.feed_new_deal(true, *px, *qty, fee);
                    }
                    order_expired.push(*oid);
                }
                // 买吃单处理
                (_, false, px, qty, 3)=>{
                    if best_ask<=*px{
                        let fee = px*qty*self.taker_fee;
                        self.position += qty;
                        pr_ref.feed_new_deal(false, *px, *qty, fee);
                    }
                    order_expired.push(*oid);
                }
                // 未成交
                _=>{}
            }
        }
        // 清除掉活跃订单区的吃单和已成交挂单
        for oid in order_expired{
            self.active_orders.remove(&oid);
        }
        // 清除掉活跃订单区的应撤挂单
        for revoke_info in self.revoking.drain(0..){
            let (oid, _, _, _, _) = revoke_info;
            self.active_orders.remove(&oid);
        }
        // 查看当前利润统计bin盈利情况:如果交易量累计满了一个bin,获取profitrecorder的累积利润水平,并重设bin容器 
        let level = pr_ref.try_eval_bin(fair);
        if let Some(lv) = level{
            self.current_profit_level = lv;
            pr_ref.reset_bin(6.0*1.0, fair, self.position);
        }         
    }
}

