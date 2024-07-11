pub struct ProfitRecorder
{
    max_level:usize,
    bin_volume:f64,
    pub current_level:usize,
    total_buy_qty:f64,
    total_sell_qty:f64,
    total_buy_amt:f64,
    total_sell_amt:f64,
    bin_begin_qty:f64,
    bin_begin_fair:f64,
}

impl ProfitRecorder
{
    pub fn new(max_level:usize, bin_volume:f64, position:f64, fair:f64)->Self{
        ProfitRecorder{
            max_level:max_level,
            current_level:1,
            total_buy_amt:0.0,
            total_buy_qty:0.0,
            total_sell_amt:0.0,
            total_sell_qty:0.0,
            bin_volume:bin_volume,
            bin_begin_qty:position,
            bin_begin_fair:fair,
        }
    }

    /**
     * 在首次启用统计时, 或real_trader开启新一轮实盘盈利统计时, 先调用这个重置统计数据. 
    */
    pub fn reset_all(&mut self, bin_volume:f64, fair:f64, position:f64){
        self.reset_bin(bin_volume, fair, position);
        self.current_level = 1;
    }

    /**
     * 重置统计的容器bin
    */
    pub fn reset_bin(&mut self, bin_volume:f64, fair:f64, position:f64){
        self.bin_volume = bin_volume;
        (self.total_buy_amt, self.total_buy_qty, self.total_sell_amt, self.total_sell_qty) = (0.0,0.0,0.0,0.0);
        self.bin_begin_qty = position;
        self.bin_begin_fair = fair;
    }


    /**
     * 试结算最近一个bin的盈亏.
     * 如果bin的容量未满, 则返回当前level;
     * 如果bin已满, 则返回盈亏数字.
    */
    pub fn try_eval_bin(&mut self, fair:f64)->Option<usize>{
        if self.total_buy_qty+self.total_sell_qty>=self.bin_volume{
            // profit不仅包含交易所得, 也包含持仓所得.
            let profit = (self.total_buy_qty-self.total_sell_qty)*fair + self.total_sell_amt - self.total_buy_amt + self.bin_begin_qty*(fair-self.bin_begin_fair);
            if profit>0.0{
                self.current_level = self.max_level.min(self.current_level+1);
            }else{
                self.current_level = 1.max(self.current_level-1);
            }
            Some(self.current_level)
        }else{
            None
        }
    }

    /**
     * 汇集账户状态.
     * 当trade量超过指定统计量时:
     *  1、统计该交易周期内的盈亏
     *  2、调整累计盈利水平，并向外部通知这个水平;
     *  3、从外部获取下一轮统计量大小
    */
    pub fn feed_new_deal(&mut self, is_sell:bool, price:f64, qty:f64, fee:f64){

        if is_sell{
            self.total_sell_qty += qty;
            self.total_sell_amt += qty*price - fee;
        }else{
            self.total_buy_qty += qty;
            self.total_buy_amt += qty*price + fee;
        }
            
    }
}