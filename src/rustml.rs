pub struct RustML;
impl RustML {
    fn sum(x: &[i32]) -> i32 {
        x.iter().sum()
    }

    fn exp(x: i32) -> i32 {
        f64::exp(x as f64) as i32
    }

    pub fn Sigmod(x: i32) -> i32 {
        1 / (1 + Self::exp(-x)) 
    }

    pub fn Softmax(x: &[i32]) -> f64 {
        // Apply exp to each element and then sum the results
        let exp_x: Vec<i32> = x.iter().map(|&v| Self::exp(v)).collect();
        let sum_exp_x = Self::sum(&exp_x);

        // Return the Softmax result as a float (f64) to avoid integer division
        exp_x.iter().map(|&v| v as f64 / sum_exp_x as f64).sum()
    }

    pub fn ReLU(x: i32) -> i32 {
        std::cmp::max(0, x)
    }

    pub fn RNN(&self, x: &[i32], y: &[i32], weights: &[i32], bias: i32) -> Vec<i32> {
        let mut yt: Vec<i32> = Vec::new();

        // ht (hidden state) calculation using Sigmoid
        for i in 0..x.len() {
            let temp_value: i32 = (x[i] * weights[i]) + y[i] + bias;
            yt.push(Self::Sigmod(temp_value)); 
        }

        yt
    }
}