pub enum ValueFunctionEnum {
    LinearRegression,
}

impl std::fmt::Display for ValueFunctionEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ValueFunctionEnum::LinearRegression => {
                write!(f, "LinearRegression")
            }
        }
    }
}
