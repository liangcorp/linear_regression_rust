//! Gradient descent with multiple features
//!
//! This crate is a collection of functions to perform
//! calculation of gradient descent

use std::io;
use std::io::{Error, ErrorKind};

/// # Linear Regression Hypothesis
/// h(x) = theta_0 * x_0 + theta_1 * x_1
fn linear_regression_hypothesis(
    x_mtrx: &[Vec<f32>],
    theta: &mut [f32],
    num_feat: usize,
    num_train: usize,
) -> Vec<f32> {
    let mut h_x = vec![0.0; num_train];
    let mut sum;

    for (i, x_row) in x_mtrx.iter().enumerate().take(num_train) {
        sum = 0.0;
        for j in 0..num_feat {
            sum += theta[j] * x_row[j];
        }

        h_x[i] = sum;
    }

    h_x
}

/// # Gradient descent
///
/// - X and y are the training sets.
/// - alpha is the learning rate
/// - theta is a chosen number.
///
/// ## Implement the following matlab formula:
///
/// theta(indx,:) = theta(indx,:) -
///                 alpha * ((((temp[] * X[]) - y[]) * X(:,indx))/m);
pub fn get_thetas(
    x_mtrx: &[Vec<f32>],
    y_vec: &[f32],
    alpha: f32,
    theta: &mut [f32],
    iterations: u32,
) -> Result<Vec<f32>, io::Error> {
    let num_train = y_vec.len(); // no of training sets
    let num_feat = theta.len();

    let mut sum: f32;
    let mut h_x = vec![0.0; num_train];

    if x_mtrx.len() != num_train {
        return Err(Error::new(ErrorKind::Other, "Mis-matching training sets"));
    }

    // Convert &[Vec<f32>] to &[&[f32]]
    // to speeds up the execution by a little
    let mut x_vec_slice: Vec<&[f32]> = Vec::with_capacity(num_train);

    for x_row in x_mtrx.iter().take(num_train) {
        x_vec_slice.push(&x_row[..]);
    }

    let x_slice = &x_vec_slice[..];

    for _ in 0..iterations {
        // Shadow h_x from vec to vec slice to speeds up the execution a bit
        let h_x = &mut h_x[..];
        // Linear Regression Hypothesis
        // h(x) = theta_0 * x_0 + theta_1 * x_1
        for (i, x_row) in x_slice[..].iter().enumerate().take(num_train) {
            sum = 0.0;
            for j in 0..num_feat {
                sum += theta[j] * x_row[j];
            }

            h_x[i] = sum;
        }

        for (j, t) in theta.iter_mut().enumerate().take(num_feat) {
            sum = 0.0;

            for i in 0..num_train {
                sum += (h_x[i] - y_vec[i]) * x_slice[i][j];
            }

            *t -= alpha * sum / num_train as f32;
        }
    }

    Ok(theta.to_vec())
}

/// # Gradient descent
///
/// - X and y are the training sets.
/// - alpha is the learning rate
/// - theta is a chosen number.
///
/// ## Implement the following matlab formula:
///
/// theta(indx,:) = theta(indx,:) -
///                 alpha * ((((temp[] * X[]) - y[]) * X(:,indx))/m);
pub fn get_thetas_hypothesis_function(
    x_mtrx: &[Vec<f32>],
    y_vec: &[f32],
    alpha: f32,
    theta: &mut [f32],
    iterations: u32,
) -> Result<Vec<f32>, io::Error> {
    let num_train = y_vec.len(); // no of training sets
    let num_feat = theta.len();

    let mut sum: f32;

    if x_mtrx.len() != num_train {
        return Err(Error::new(ErrorKind::Other, "Mis-matching training sets"));
    }

    for _ in 0..iterations {
        // Shadow h_x from vec to vec slice
        // speeds up the execution a bit
        let h_x = linear_regression_hypothesis(x_mtrx, theta, num_feat, num_train);

        for (j, t) in theta.iter_mut().enumerate().take(num_feat) {
            sum = 0.0;

            for i in 0..num_train {
                sum += (h_x[i] - y_vec[i]) * x_mtrx[i][j];
            }

            *t -= alpha * sum / num_train as f32;
        }
    }

    Ok(theta.to_vec())
}

/// # Gradient descent
///
/// - X and y are the training sets.
/// - alpha is the learning rate
/// - theta is a chosen number.
///
/// ## Implement the following matlab formula:
///
/// theta(indx,:) = theta(indx,:) -
///                 alpha * ((((temp[] * X[]) - y[]) * X(:,indx))/m);
pub fn get_thetas_flatten_x(
    flattened_x: &[f32],
    y_vec: &[f32],
    alpha: f32,
    num_feat: usize,
    theta: &mut [f32],
    iterations: u32,
) -> Result<Vec<f32>, io::Error> {
    let num_train = y_vec.len(); // no of training sets

    let mut sum: f32;
    let mut k;

    for _ in 0..iterations {
        let mut h_x = Vec::with_capacity(num_train);
        let mut i = 0;

        while i < num_train * num_feat {
            sum = 0.0;
            for j in 0..num_feat {
                sum += theta[j] * flattened_x[i + j];
            }

            h_x.push(sum);
            i += num_feat;
        }

        for j in 0..num_feat {
            sum = 0.0;
            k = 0;

            for i in 0..num_train {
                sum += (h_x[i] - y_vec[i]) * flattened_x[k + j];
                k += num_feat;
            }

            theta[j] -= alpha * sum / num_train as f32;
        }
    }

    Ok(theta.to_vec())
}
