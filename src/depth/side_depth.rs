use hftbacktest::depth::{INVALID_MAX, INVALID_MIN};

use super::{LobMatrix, MAX_DEPTH};


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
    pub fn best_item(&self) -> (i32,i32) {
        let item = self.depth.row(self.best_index as usize);
        return (item[0], item[1]);
    }

    pub fn best_tick(&self) -> i32 {
        return self.best_item().0;
    }

    pub fn best_price(&self) -> f32 {
        self.best_tick() as f32 * self.tick_size
    }
    
    pub fn best_qty_lot(&self) -> i32 {
        return self.best_item().1;
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