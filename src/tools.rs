use ringbuf::{traits::{Consumer, Observer, Producer}, HeapRb};
use std::fmt::Debug;

pub fn push_ring_buffer<T:Debug>(rb: &mut HeapRb<T>, value: T)->Option<T>{
    let mut ele: Option<T> = None;
    if rb.is_full() {
        ele = Some(rb.try_pop().unwrap());
    }
    rb.try_push(value).unwrap();
    ele
}