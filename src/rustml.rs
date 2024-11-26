use rand::Rng;
use libm::tanh;


pub struct RustML;
impl RustML {
    fn sum(x: &[f64]) -> f64 {
        x.iter().sum()
    }

    fn exp(x: f64) -> f64 {
        f64::exp(x as f64) as f64
    }

    pub fn Sigmoid(x: f64) -> f64 {    
        1.0 / (1.0 + Self::exp(-x)) 
    }

    pub fn Softmax(x: &[f64]) -> f64 {
        // Apply exp to each element and then sum the results
        let exp_x: Vec<f64> = x.iter().map(|&v| Self::exp(v)).collect();
        let sum_exp_x = Self::sum(&exp_x);

        // Return the Softmax result as a float (f64) to avoid integer division
        exp_x.iter().map(|&v| v as f64 / sum_exp_x as f64).sum()
    }

    pub fn ReLU(x: f64) -> f64 {
        x.max(0.0)
    }

    pub fn RNN(&self, x: &[f64], y: &[f64], weights: &[f64], bias: f64) -> Vec<f64> {
        let mut yt: Vec<f64> = Vec::new();

        // ht (hidden state) calculation using Sigmoid
        for i in 0..x.len() {
            let temp_value: f64 = (x[i] * weights[i]) + y[i] + bias;
            yt.push(Self::Sigmoid(temp_value)); 
        }

        yt
    }

    pub fn LSTM(&self, x: &[f64], hx: &[f64], weights: &[f64]) -> Vec<f64> {
        let mut yt: Vec<f64> = Vec::new();

        for i in 0..x.len() {
            let ftValue = rand::thread_rng().gen_range(-1.0..1.0); 
            let ctAndhtValue = rand::thread_rng().gen_range(-1.0..1.0); 

            if yt.len() < x.len() {
                // Ensure `yt` has enough elements by initializing it
                yt.resize(x.len(), 0.0); // Resize to the length of `x` and fill with 0.0
            }

            if yt.len() == 0 {
                let ft: f64 = Self::Sigmoid((weights[i] * (hx[i] - 1.0)) + ftValue);
                let ct: f64 = (ft * (weights[i] - 1.0) + 1.0) + ctAndhtValue; 
                yt.push((ft * tanh(ct.into())) + ctAndhtValue); 
            } else {
                let ft: f64 = Self::Sigmoid((weights[i] * (hx[i] - 1.0)) + ftValue);
                let ct: f64 = (ft * (weights[i] - 1.0) + 1.0) + ctAndhtValue; 
                yt[i] -= (ft * tanh(ct.into())) + ctAndhtValue; 
            }
            
        }

        yt
    }
}