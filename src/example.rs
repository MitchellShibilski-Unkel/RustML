mod rustml;

fn main() {
    let RML = rustml::RustML;
    let test: [i32; 4] = [1, 2, 3, 4];
    let testy: [i32; 4] = [5, 6, 7, 8];
    let testw: [i32; 4] = [55, 66, 77, 88];
    println!("{:?}", RML.RNN(&test, &testy, &testw, 1));
}
