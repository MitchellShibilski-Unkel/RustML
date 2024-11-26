# RustML
Open-source AI Library for Rust

## About
`RustML` is an AI/machine learning library built for the `Rust` programming language. It's desgined to work similarly to `Python` libraries, such as PyTorch & Tensorflow. 

## Current Functions & Features
- Sigmod function
```rust
    Sigmod(x: i32) -> i32
```

- ReLU function
```rust
    ReLU(x: i32) -> i32
```

- Softmax function
```rust
    Softmax(x: &[i32]) -> f64
```

- RNN algoritm
```rust
    RNN(x: &[i32], y: &[i32], weights: &[i32], bias: i32) -> Vec<i32> 
```