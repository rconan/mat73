//! Import Matlab 7.3 mat files array into Rust
//!
//! # Examples
//! Writing a Matlab array into a `Vec`:
//!```
//! let file = "examples/arrays.mat";
//! let mat_file = mat73::File::new(file).unwrap();
//! let var: Vec<f64> = mat_file.array("q").unwrap().into();
//!```
//! Writing a Matlab array into a [nalgebra](https://crates.io/crates/nalgebra) matrix:
//!```
//! let file = "examples/arrays.mat";
//! let mat_file = mat73::File::new(file).unwrap();
//! let var: nalgebra::DMatrix<f64> = mat_file.array("w").unwrap().into();
//!```

use hdf5::types::FixedAscii;

pub enum Error {
    HDF5(hdf5::Error),
    Dataset(String),
    Group(String),
    Struct,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::HDF5(e) => write!(
                f,
                "Importing Matlab variables failed ({}), caused by {}",
                e,
                <Self as std::error::Error>::source(self).unwrap()
            ),
            Error::Dataset(name) => write!(f, "Loading {} dataset failed", name),
            Error::Group(name) => write!(f, "Loading {} group failed", name),
            Error::Struct => write!(f, "Matlab class is not a struct"),
        }
    }
}
impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Error as std::fmt::Display>::fmt(self, f)
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self {
            Error::HDF5(e) => e.source(),
            _ => None,
        }
    }
}
impl From<hdf5::Error> for Error {
    fn from(error: hdf5::Error) -> Self {
        Error::HDF5(error)
    }
}
pub type Result<T> = ::std::result::Result<T, Error>;

/// Matlab variables
#[derive(Debug)]
pub struct MatVar<T> {
    name: String,
    shape: Vec<usize>,
    data: Vec<T>,
}
impl<T> MatVar<T> {
    pub fn raw(self) -> Vec<T> {
        self.data
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn n_row(&self) -> usize {
        self.shape[1]
    }
    pub fn n_column(&self) -> usize {
        self.shape[0]
    }
}
/// Matlab struct
#[derive(Debug)]
pub struct MatStruct {
    pub field_names: Vec<String>,
}

/// Matlab 7.3 mat file
pub struct File {
    h5: hdf5::File,
}
impl File {
    /// Open a Matlab mat file
    pub fn new<P: AsRef<std::path::Path>>(file: P) -> Result<Self> {
        Ok(Self {
            h5: hdf5::File::open(file)?,
        })
    }
    /// Read a Matlab array
    pub fn array<T: hdf5::H5Type>(&self, name: &str) -> Result<MatVar<T>> {
        let dataset = match self.h5.dataset(name) {
            Ok(it) => it,
            _ => return Err(Error::Dataset(name.to_string())),
        };
        Ok(MatVar {
            name: dataset.name(),
            shape: dataset.shape(),
            data: dataset.read_raw::<T>()?,
        })
    }
    pub fn structure(&self, name: &str) -> Result<MatStruct> {
        let group = match self.h5.group(name) {
            Ok(it) => it,
            _ => return Err(Error::Group(name.to_string())),
        };
        group
            .attr("MATLAB_class")?
            .read_scalar::<hdf5::types::FixedAscii<256>>()
            .and_then(|matlab_type| {
                if matlab_type.as_str() == "struct" {
                    Ok(())
                } else {
                    Err("Matlab class is not a struct".into())
                }
            })?;
        let field_names = group.attr("MATLAB_fields")?.read_raw().map(
            |data: Vec<hdf5::types::VarLenArray<FixedAscii<1>>>| {
                data.into_iter()
                    .map(|v| v.iter().map(|x| x.as_str()).collect::<String>())
                    .collect::<Vec<String>>()
            },
        )?;
        Ok(MatStruct { field_names })
    }
}

/// Creates a Rust `Vec` from a Matlab array, column wise
impl<T> From<MatVar<T>> for Vec<T> {
    fn from(var: MatVar<T>) -> Self {
        var.raw()
    }
}
#[cfg(feature = "nalgebra")]
/// Creates a nalgebra matrix from a Matlab 2D array
impl<T: 'static + std::marker::Copy + std::cmp::PartialEq + std::fmt::Debug> From<MatVar<T>>
    for nalgebra::Matrix<
        T,
        nalgebra::Dynamic,
        nalgebra::Dynamic,
        nalgebra::VecStorage<T, nalgebra::Dynamic, nalgebra::Dynamic>,
    >
{
    fn from(var: MatVar<T>) -> Self {
        let shape = var.shape();
        if shape.len() > 2 {
            unimplemented!("Matlab array dimension cannot be greater than 2")
        } else {
            nalgebra::DMatrix::from_column_slice(shape[1], shape[0], &var.raw())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn array_to_vec() {
        let file = "examples/arrays.mat";
        let mat_file = File::new(file).unwrap();
        let var: Vec<f64> = mat_file.array("q").unwrap().into();
        assert_eq!(var, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    }
    #[test]
    fn array_to_matrix() {
        let file = "examples/arrays.mat";
        let mat_file = File::new(file).unwrap();
        let var: nalgebra::DMatrix<f64> = mat_file.array("w").unwrap().into();
        assert_eq!(
            var,
            nalgebra::DMatrix::from_column_slice(3, 2, &[0., -6., -2., 1., 56., 1.])
        )
    }
}
