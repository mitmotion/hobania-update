/// Used to represent one of many possible errors that may be omitted by the
/// rendering subsystem.
pub enum RenderError {
    RequestDeviceError(wgpu::RequestDeviceError),
    MappingError(wgpu::BufferAsyncError),
    SwapChainError(wgpu::SwapChainError),
    CustomError(String),
    CouldNotFindAdapter,
    GlslIncludeError(String, glsl_include::Error),
    ParserError(String, naga::front::glsl::ParseError),
    ValidationError(String, naga::valid::ValidationError),
    SpirvError(String, naga::back::spv::Error),
}

use std::fmt;
impl fmt::Debug for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestDeviceError(err) => {
                f.debug_tuple("RequestDeviceError").field(err).finish()
            },
            Self::MappingError(err) => f.debug_tuple("MappingError").field(err).finish(),
            Self::SwapChainError(err) => f
                .debug_tuple("SwapChainError")
                // Use Display formatting for this error since they have nice descriptions
                .field(&format!("{}", err))
                .finish(),
            Self::CustomError(err) => f.debug_tuple("CustomError").field(err).finish(),
            Self::CouldNotFindAdapter => f.debug_tuple("CouldNotFindAdapter").finish(),
            Self::GlslIncludeError(shader_name, err) => write!(
                f,
                "\"{}\" shader contains invalid include directives: {}",
                shader_name, err
            ),
            Self::ParserError(shader_name, err) => write!(
                f,
                "\"{}\" shader failed to parse due to the following error: {}",
                shader_name, err
            ),
            Self::ValidationError(shader_name, err) => write!(
                f,
                "\"{}\" shader failed to validate due to the following error: {}",
                shader_name, err
            ),
            Self::SpirvError(shader_name, err) => write!(
                f,
                "\"{}\" shader failed to emit due to the following error: {}",
                shader_name, err
            ),
        }
    }
}

impl From<wgpu::RequestDeviceError> for RenderError {
    fn from(err: wgpu::RequestDeviceError) -> Self { Self::RequestDeviceError(err) }
}

impl From<wgpu::BufferAsyncError> for RenderError {
    fn from(err: wgpu::BufferAsyncError) -> Self { Self::MappingError(err) }
}

impl From<wgpu::SwapChainError> for RenderError {
    fn from(err: wgpu::SwapChainError) -> Self { Self::SwapChainError(err) }
}

impl From<(&str, glsl_include::Error)> for RenderError {
    fn from((shader_name, err): (&str, glsl_include::Error)) -> Self {
        Self::GlslIncludeError(shader_name.into(), err)
    }
}

impl From<(&str, naga::front::glsl::ParseError)> for RenderError {
    fn from((shader_name, err): (&str, naga::front::glsl::ParseError)) -> Self {
        Self::ParserError(shader_name.into(), err)
    }
}
impl From<(&str, naga::valid::ValidationError)> for RenderError {
    fn from((shader_name, err): (&str, naga::valid::ValidationError)) -> Self {
        Self::ValidationError(shader_name.into(), err)
    }
}
impl From<(&str, naga::back::spv::Error)> for RenderError {
    fn from((shader_name, err): (&str, naga::back::spv::Error)) -> Self {
        Self::SpirvError(shader_name.into(), err)
    }
}
