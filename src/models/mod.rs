use nalgebra::DMatrix;

pub mod analyticmarketdepth;
pub mod distance_model;

pub use analyticmarketdepth::AnalyticMarketDepth;
pub use distance_model::DistanceProfitResult;

const MAX_DEPTH: i32 = 5000_i32;
type EvaluateMatrix = DMatrix<f32>;  // 5000*5的f32矩阵,列分别为price, qty, hit_prob, exp_rtn, exp_hit_time 