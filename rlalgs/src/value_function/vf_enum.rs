pub enum ValueFunctionEnum {
    LinearRegression { step_size: f32 },
}

impl std::fmt::Display for ValueFunctionEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ValueFunctionEnum::LinearRegression { step_size } => write!(
                f,
                "LinearRegression with step_size: {step_size}"
            )
        }
        
    }
}
