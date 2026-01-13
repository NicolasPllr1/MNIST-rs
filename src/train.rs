use ndarray::Array1;

use crate::{Module, NN};
use std::fs;
use std::path::Path;

type CostFunction = fn(expected_y: &Array1<f32>, actual_y: &Array1<f32>) -> Array1<f32>;

fn least_squares(expected_y: &Array1<f32>, actual_y: &Array1<f32>) -> Array1<f32> {
    expected_y - actual_y.mapv(|x| x * x)
}

trait Optimizer {
    fn step(&self, nn: &NN, cost_function: CostFunction);
}

struct SGD {
    learning_rate: f32,
}

impl Optimizer for SGD {
    fn step(&self, nn: &NN, cost_function: CostFunction) {
        let num_layers = nn.layers.len();
        // TODO: Initialize prev_grad properly
        let mut prev_grad = cost_function(???);
        for layer in (0..num_layers).rev() {
            prev_grad = nn.layers[layer].backward(prev_grad);
        }

    }
}

/// Train a neural network (stub)
pub fn train(
    train_steps: usize,
    learning_rate: f32,
    checkpoint_folder: &str,
    checkpoint_stride: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create checkpoint folder if it doesn't exist
    fs::create_dir_all(checkpoint_folder)?;

    // TODO: Implement training logic
    // For now, use a hardcoded neural network
    let mut nn = NN { layers: vec![] };

    // Original main() code from before rebase:
    /*
    fn main() {
        let mut nn = NN {
            layers: vec![
                Layer::FC(FcLayer::new(784, 10)),
            ],
        };

        let (train_images, train_labels, test_images, test_labels) = load_mnist();
        println!("Train images: {:?}", train_images.len());
        println!("Train labels: {:?}", train_labels.len());
        println!("Test images: {:?}", test_images.len());
        println!("Test labels: {:?}", test_labels.len());

        // train loop
        // TODO: calculate loss and do backpropagation
        println!("Training...");
        for (image, _label) in train_images.chunks(784).zip(train_labels.iter()).progress_count(60_000) {
            let img_f32: Vec<f32> = image.iter().map(|&x| x as f32).collect();
            let input = Array2::from_shape_vec((1, 784), img_f32).unwrap();
            let _output = nn.forward(input);
        }

        // test loop
        // TODO: calculate accuracy
        println!("Testing...");
        let mut total_correct = 0;
        let mut total_samples = 0;
        for (image, label) in test_images.chunks(784).zip(test_labels.iter()).progress_count(10_000) {
            let img_f32: Vec<f32> = image.iter().map(|&x| x as f32).collect();
            let input = Array2::from_shape_vec((1, 784), img_f32).unwrap();
            let output = nn.forward(input);
            // Find index of max value (argmax)
            let predicted_label = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap() as u8;
            if predicted_label == *label {
                total_correct += 1;
            }
            total_samples += 1;
        }
        let accuracy = total_correct as f32 / total_samples as f32;
        println!("Accuracy: {:?}", accuracy);
    }
    */

    let optimizer = SGD { learning_rate };

    for step in 0..train_steps {

        // Run forward pass on the dataloader?

        optimizer.step(&nn, least_squares);

        // Save checkpoint every checkpoint_stride steps
        if (step + 1) % checkpoint_stride == 0 {
            let checkpoint_num = (step + 1) / checkpoint_stride;
            let checkpoint_path =
                Path::new(checkpoint_folder).join(format!("checkpoint_{}.json", checkpoint_num));
            nn.to_checkpoint(checkpoint_path.to_str().unwrap())?;
            println!("Saved checkpoint {} at step {}", checkpoint_num, step + 1);
        }
    }

    println!("Training completed!");
    println!("Checkpoint folder: {}", checkpoint_folder);
    Ok(())
}
