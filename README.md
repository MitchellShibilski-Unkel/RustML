# RustML
Open-source AI Library for Rust
__v1.1__

## About
`RustML` is an AI/machine learning library built for the `Rust` programming language. It's desgined to work similarly to `Python` libraries, such as PyTorch & Tensorflow. 

## Current Functions & Features
- Sigmoid function
```rust
Sigmoid(x: f64) -> f64
```

- ReLU function
```rust
ReLU(x: f64) -> f64
```

- Softmax function
```rust
Softmax(x: &[f64]) -> f64
```

- RNN algoritm
```rust
RNN(x: &[f64], y: &[f64], weights: &[f64], bias: f64) -> Vec<f64> 
```

- LSTM algoritm
```rust
LSTM(x: &[f64], y: &[f64], weights: &[f64]) -> Vec<f64> 
```