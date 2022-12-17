use std::ops::{Add, Mul};

/// A multi-dimensional array of floating-point values.
struct Tensor {
    /// The values of the tensor as a vector of floating-point numbers.
    data: Vec<f32>,
    /// The size of each dimension of the tensor as a vector of unsigned integers.
    shape: Vec<usize>,
}

impl Tensor {
    /// Creates a new tensor with the given `data` and `shape`.
    fn new(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor { data, shape }
    }

    /// Creates a new tensor with the given `shape` and all values initialized to 0.
    fn from_shape(shape: Vec<usize>) -> Tensor {
        let size = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape,
        }
    }

    /// Reshapes the tensor to have the given `new_shape`.
    ///
    /// # Panics
    ///
    /// Panics if the new shape has a different size than the original shape.
    fn reshape(&mut self, new_shape: Vec<usize>) {
        let size: usize = self.shape.iter().product();
        let new_size = new_shape.iter().product();
        if size != new_size {
            panic!("Cannot reshape tensor of size {} to size {}", size, new_size);
        }
        self.shape = new_shape;
    }

    /// Returns the size of the tensor, i.e. the total number of elements.
    fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns a reference to the shape of the tensor.
    fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    /// Returns the rank of the tensor, i.e. the number of dimensions.
    fn rank(&self) -> usize {
        self.shape.len()
    }
}

impl Add for Tensor {
    type Output = Tensor;

    /// Adds two tensors element-wise and returns the result.
    ///
    /// # Panics
    ///
    /// Panics if the two tensors have different sizes.
    fn add(self, other: Tensor) -> Tensor {
        let size = self.size();
        if size != other.size() {
            panic!("Cannot add tensors of different sizes ({} and {})", size, other.size());
        }
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(self.data[i] + other.data[i]);
        }
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }
}

/// Multiplies two tensors element-wise and returns the result.
///
/// # Panics
///
/// Panics if the two tensors have different sizes.
impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        let size = self.size();
        if size != other.size() {
            panic!("Cannot multiply tensors of different sizes ({} and {})", size, other.size());
        }
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(self.data[i] * other.data[i]);
        }
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_tensor() {
        use super::Tensor;
        let mut t = Tensor::from_shape(vec![2, 3]);
        assert_eq!(t.size(), 6);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.shape(), &vec![2, 3]);
        t.reshape(vec![3, 2]);
        assert_eq!(t.size(), 6);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.shape(), &vec![3, 2]);
    }

    #[test]
    fn test_add() {
        use super::Tensor;
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let t3 = t1 + t2;
        assert_eq!(t3.data, vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(t3.shape, vec![2, 2]);
    }

    #[test]
    fn test_mul() {
        use super::Tensor;
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let t3 = t1 * t2;
        assert_eq!(t3.data, vec![5.0, 12.0, 21.0, 32.0]);
        assert_eq!(t3.shape, vec![2, 2]);
    }
}