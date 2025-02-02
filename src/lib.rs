//! Implementation of linear regression
use std::path::Path;
// use std::time::Instant;

pub mod cost_functions;
pub mod gradient_descent;
pub mod normal_equation;
pub mod read_data;

const ITERATIONS: u32 = 5000; // the learning speed

// Sample run of linear regression
pub fn sample_run(input_file_path: &Path) {
    let (x, y) = match read_data::get_data(input_file_path) {
        Ok((x, y)) => (x, y),
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    };

    let mut theta = vec![0.0; x[0].len()]; // set theta 0 and theta 1 to 0.0

    // set the learning rate = no of features / 10
    let alpha = if x[0].len() < 3 {
        0.01
    } else {
        x[0].len() as f32 / 10.0
    };

    match gradient_descent::get_thetas(&x, &y, alpha, &mut theta, ITERATIONS) {
        Ok(theta) => {
            println!("Found thetas using Gradient Descent with learning speed {} and {} number of iterations: {:?}", alpha, ITERATIONS, &theta[1..]);
        }
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }

    // match gradient_descent::get_thetas_hypothesis_function(&x, &y, alpha, &mut theta, ITERATIONS) {
    //     Ok(theta) => {
    //         println!("Hypothesis function");
    //         println!("Found thetas using Gradient Descent with learning speed {} and {} number of iterations: {:?}", alpha, ITERATIONS, &theta[1..]);
    //     }
    //     Err(e) => panic!("{}", e.get_ref().unwrap()),
    // }

    // Flattened X
    // Read data from file
    // let (flattened_x, num_feat, y) = match read_data::get_data_flat_x(input_file_path) {
    //     Ok((flattened_x, num_feat, y)) => (flattened_x, num_feat, y),
    //     Err(e) => panic!("{}", e.get_ref().unwrap()),
    // };
    //
    // // set the learning rate = no of features / 10
    // let alpha = if num_feat < 3 {
    //     0.01
    // } else {
    //     num_feat as f32 / 10.0
    // };
    //
    // let mut theta = vec![0.0; num_feat]; // set theta 0 and theta 1 to 0.0
    // match gradient_descent::get_thetas_flatten_x(
    //     &flattened_x,
    //     &y,
    //     alpha,
    //     num_feat,
    //     &mut theta,
    //     ITERATIONS,
    // ) {
    //     Ok(theta) => {
    //         println!(
    //             "Gradient Descent with learning speed {} and {} number of iterations:\n {:?}",
    //             alpha,
    //             ITERATIONS,
    //             &theta[1..]
    //         );
    //     }
    //     Err(e) => panic!("{}", e.get_ref().unwrap()),
    // }

    // match normal_equation::get_theta(&x, &y) {
    //     Ok(theta) => {
    //         print!("Found thetas using Normal Equation (skipping theta 0): [");
    //         for t in theta.iter().skip(1) {
    //             print!(" {} ", t);
    //         }
    //         println!("]");
    //     }
    //     Err(e) => panic!("{}", e.get_ref().unwrap()),
    // }

    // match cost_functions::get_cost(&x, &y, &theta) {
    //     Ok(j_theta) => {
    //         println!("Thetas are {:?}, J(theta) is {:?}", theta, j_theta);
    //     }
    //     Err(e) => eprint!("{}", e.get_ref().unwrap()),
    // }
}
#[cfg(test)]
mod tests {
    use super::linear_regression::normal_equation;

    #[test]
    #[should_panic]
    fn three_by_four() {
        let matrix = vec![
            vec![1.0, 1.0, 1.0, -1.0],
            vec![1.0, 1.0, -1.0, 1.0],
            vec![1.0, -1.0, 1.0, 1.0],
        ];

        match normal_equation::get_determinant(&matrix) {
            Ok(der) => assert_eq!(der, -16_f64),
            Err(e) => panic!("{}", e.get_ref().unwrap()),
        };
    }

    #[test]
    fn four_by_four() {
        let matrix = vec![
            vec![1.0, 1.0, 1.0, -1.0],
            vec![1.0, 1.0, -1.0, 1.0],
            vec![1.0, -1.0, 1.0, 1.0],
            vec![-1.0, 1.0, 1.0, 1.0],
        ];

        match normal_equation::get_determinant(&matrix) {
            Ok(der) => assert_eq!(der, -16_f64),
            Err(e) => panic!("{}", e.get_ref().unwrap()),
        };
    }
}
