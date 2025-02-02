pub use std::path::Path;

use linear_regression::sample_run;
use std::{env, io};

const ARGS_SIZE_LIMIT: usize = 2;

fn argument_check(args: &[String]) -> Result<(), io::Error> {
    let mut error = String::new();

    if args.len() < ARGS_SIZE_LIMIT {
        error = String::from("not enough input argument");
    } else if args[1].is_empty() {
        error = String::from("filename is empty");
    }

    if error.is_empty() {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::NotFound, error))
    }
}

fn display_help(err: io::Error) {
    eprintln!("ERROR: {}\n", err);
    let help_message = String::from("Usage: Sample run using input data file");
    println!("{}", help_message);
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // check the input arguments
    match argument_check(&args) {
        Ok(_) => println!("Reading data from path: {}", &args[1]),
        Err(e) => display_help(e),
    };

    let data_file_path = Path::new(&args[1]);

    // sample run of linear regression with data file
    sample_run(data_file_path);
}
