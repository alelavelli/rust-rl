use ndarray::{Array1, Array2, ArrayBase, Axis, Dim, ViewRepr};

use super::{PreprocessingError, Preprocessor};

/// Scale data with z-score
///
/// $$ \tilde{x} = \frac{x - \mu}{\sigma}$$
pub struct ZScore {
    means: Option<Array1<f32>>,
    stds: Option<Array1<f32>>,
}

impl Default for ZScore {
    fn default() -> Self {
        Self::new()
    }
}

impl ZScore {
    pub fn new() -> ZScore {
        ZScore {
            means: None,
            stds: None,
        }
    }
}

impl Preprocessor<f32> for ZScore {
    fn fit(
        &mut self,
        x: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Result<&mut Self, PreprocessingError> {
        self.means = Some(x.mean_axis(Axis(0)).ok_or(PreprocessingError::FitError)?);
        self.stds = Some(x.std_axis(Axis(0), 1.0));
        Ok(self)
    }

    fn transform(
        &self,
        x: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Result<Array2<f32>, PreprocessingError> {
        if let (Some(mu), Some(sigma)) = (self.means.as_ref(), self.stds.as_ref()) {
            Ok((x - mu) / sigma)
        } else {
            Err(PreprocessingError::TransformError)
        }
    }
}

/// Normalize data in the interval [a, b]
///
/// $$ \tilde{x} = a + \frac{(x - x_{min}) \times (a - b) }{x_{max} - x_{min}} $$
pub struct RangeNorm {
    a: f32,
    b: f32,
    x_min: Option<Array1<f32>>,
    x_max: Option<Array1<f32>>,
}

impl Default for RangeNorm {
    fn default() -> Self {
        Self::new(Some(0.0), Some(1.0))
    }
}

impl RangeNorm {
    pub fn new(a: Option<f32>, b: Option<f32>) -> RangeNorm {
        RangeNorm {
            a: a.unwrap_or(0.0),
            b: b.unwrap_or(1.0),
            x_min: None,
            x_max: None,
        }
    }
}

impl Preprocessor<f32> for RangeNorm {
    fn fit(
        &mut self,
        x: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Result<&mut Self, PreprocessingError> {
        self.x_min = Some(x.map_axis(Axis(0), |view| {
            *view
                .into_iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        }));
        self.x_max = Some(x.map_axis(Axis(0), |view| {
            *view
                .into_iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        }));
        Ok(self)
    }

    fn transform(
        &self,
        x: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Result<Array2<f32>, PreprocessingError> {
        if let (Some(x_min), Some(x_max)) = (self.x_min.as_ref(), self.x_max.as_ref()) {
            Ok(self.a + (x - x_min) * (self.b - self.a) / (x_max - x_min))
        } else {
            Err(PreprocessingError::TransformError)
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, Axis};
    use ndarray_rand::RandomExt;
    use rand_distr::{Normal, Uniform};

    use crate::preprocessing::Preprocessor;

    use super::{RangeNorm, ZScore};

    #[test]
    fn zscore() {
        let x = Array2::random((100, 2), Normal::new(1.0, 0.3).unwrap());
        let mut encoder = ZScore::new();
        _ = encoder.fit(&x.view());
        let x_norm = encoder.transform(&x.view()).unwrap();
        assert_abs_diff_eq!(encoder.means.unwrap(), Array1::ones(2), epsilon = 1e-1);
        assert_abs_diff_eq!(
            encoder.stds.unwrap(),
            Array1::from_elem(2, 0.3),
            epsilon = 1e-1
        );
        assert_abs_diff_eq!(
            x_norm.mean_axis(Axis(0)).unwrap(),
            Array1::zeros(2),
            epsilon = 1e-1
        );
        assert_abs_diff_eq!(
            x_norm.std_axis(Axis(0), 1.0),
            Array1::ones(2),
            epsilon = 1e-1
        );
    }

    #[test]
    fn range_norm() {
        let x = Array2::random((100, 2), Uniform::new(5.0, 10.0));
        let mut encoder = RangeNorm::default();
        _ = encoder.fit(&x.view());
        println!("{:?} {:?}", encoder.x_min, encoder.x_max);
        let x_norm = encoder.transform(&x.view()).unwrap();
        assert_abs_diff_eq!(
            x_norm.map_axis(Axis(0), |view| *view
                .into_iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()),
            Array1::zeros(2),
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            x_norm.map_axis(Axis(0), |view| *view
                .into_iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()),
            Array1::ones(2),
            epsilon = 1e-3
        );
    }
}
