mod rustml;

fn main() {
    let RML = rustml::RustML;
    let test: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let testy: [f64; 4] = [5.0, 6.0, 7.0, 8.0];
    let testw: [f64; 4] = [55.0, 66.0, 77.0, 88.0];
    println!("{:?}", RML.LSTM(&test, &testy, &testw));
}
