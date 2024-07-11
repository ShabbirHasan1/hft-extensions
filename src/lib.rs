pub mod maker_position_advisor;
pub mod models;
mod tools;
// pub use analyticmarketdepth::AnalyticMarketDepth;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub fn xx() {}

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;
    use super::*;

    #[test]
    fn it_works() {
        let a = DMatrix::<i32>::zeros(5000, 2);
        let b = DMatrix::<i32>::zeros(2, 5000);
        let c = &a * &b;
        println!("{:?}", c);
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
