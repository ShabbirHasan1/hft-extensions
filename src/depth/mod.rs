use nalgebra::DMatrix;

pub mod analyticmarketdepth;
mod side_depth;

const MAX_DEPTH: i32 = 5000_i32;
type LobMatrix = DMatrix<i32>;
type EvaluateMatrix = DMatrix<f32>;