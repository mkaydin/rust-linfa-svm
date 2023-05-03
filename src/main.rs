use std::io::Read;
use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use linfa::{Dataset, prelude::{Records, ToConfusionMatrix}, traits::{Fit, Predict, Transformer}};
use ndarray::prelude::*;
use ndarray_csv::*;
use linfa_preprocessing::linear_scaling::LinearScaler;
use linfa_svm::{Svm};

pub fn array_from_csv<R: Read>(
    csv: R,
    has_headers: bool,
    seperator: u8,

)-> Result<Array2<f64>, ReadError>{
    let mut reader = ReaderBuilder::new()
        .has_headers(has_headers)
        .delimiter(seperator)
        .from_reader(csv);

    // extract ndarray
    reader.deserialize_array2_dynamic()
}

pub fn array_from_csv_gz<R: Read>(
    gz: R,
    has_headers: bool,
    seperator: u8,
)-> Result<Array2<f64>, ReadError>{
  let file = GzDecoder::new(gz);
  array_from_csv(file, has_headers, seperator)
}

pub fn winequality() -> Dataset<f64, usize, Ix1> {
    let data = include_bytes!("../winequality-red.csv.gz");
    let array = array_from_csv_gz(&data[..],true,b',').unwrap();

    let (data, targets) = (
        array.slice(s![..,0..11]).to_owned(),
        array.column(11).to_owned(),
    );

    let feature_names = vec![
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ];

    Dataset::new(data, targets)
        .map_targets(|x| *x as usize)
        .with_feature_names(feature_names)
}

pub fn svm()->linfa_svm::error::Result<()>{
    let (train,valid) = winequality()
        .map_targets(|x| *x > 6)
        .split_with_ratio(0.7);

    println!(
        "Fit SVM classifier with #{} training points", train.nsamples()
    );


    let model = Svm::<_, bool>::params()
        .pos_neg_weights(50000., 5000.)
        .gaussian_kernel(80.0)
        .fit(&train)?;

        println!("{}", model);
        // A positive prediction indicates a good wine, a negative, a bad one
        fn tag_classes(x: &bool) -> String {
            if *x {
                "good".into()
            } else {
                "bad".into()
            }
        }
    
    let valid = valid.map_targets(tag_classes);
    
    let pred = model.predict(&valid).map(tag_classes);
       
    let cm = pred.confusion_matrix(&valid)?;
    
    println!("{:?}", cm);
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok (())
}

pub fn svm_linear_scaling()->linfa_svm::error::Result<()>{
    let (train,valid) = winequality()
        .map_targets(|x| *x > 6)
        .split_with_ratio(0.7);

    println!(
        "Fit SVM classifier with #{} training points", train.nsamples()
    );

    let scaler = LinearScaler::standard()
        .fit(&train).unwrap();
    let train_pre = scaler.transform(train);
    let valid_pre = scaler.transform(valid);

    let model_pre = Svm::<_, bool>::params()
        .pos_neg_weights(50000., 5000.)
        .gaussian_kernel(80.0)
        .fit(&train_pre)?;

    println!("{}", model_pre);
        // A positive prediction indicates a good wine, a negative, a bad one
    
        println!("{}", model_pre);
        // A positive prediction indicates a good wine, a negative, a bad one
        fn tag_classes(x: &bool) -> String {
            if *x {
                "good".into()
            } else {
                "bad".into()
            }
        }
    
    let valid_pre = valid_pre.map_targets(tag_classes);
    
    let pred_pre = model_pre.predict(&valid_pre).map(tag_classes);
           
    let cm_pre = pred_pre.confusion_matrix(&valid_pre)?;

    println!("{:?}", cm_pre);
    println!("accuracy {}, MCC {}", cm_pre.accuracy(), cm_pre.mcc());

    Ok (())

}

fn main() {

    svm();
    svm_linear_scaling();

}
