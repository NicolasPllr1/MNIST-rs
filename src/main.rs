use ndarray::prelude::*;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

trait Module {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32>; // Input is (batch_size, features)
    fn backward(&mut self, next_layer_grad: Array2<f32>) -> Array2<f32>;
}

#[derive(Debug)]
struct FcLayer {
    input_size: usize,
    output_size: usize,
    weights: Array2<f32>, // (input_size, output_size)
    bias: Array1<f32>,    //  (output_size)
    // for backprop
    last_input: Option<Array2<f32>>, // (batch_size, input_size)
    //
    w_grad: Option<Array2<f32>>, // (batch_size, input_size, output_size)
    b_grad: Option<Array2<f32>>, // (batch_size, output_size)
}

impl FcLayer {
    fn new(input_size: usize, output_size: usize) -> FcLayer {
        FcLayer {
            input_size,
            output_size,
            weights: FcLayer::init_2d_mat(input_size, output_size),
            bias: FcLayer::init_bias(output_size),
            //
            last_input: None,
            //
            w_grad: None,
            b_grad: None,
        }
    }
    fn init_bias(output_size: usize) -> Array1<f32> {
        return Array1::random(output_size, Uniform::new(0., 1.0).unwrap());
    }
    fn init_2d_mat(input_size: usize, output_size: usize) -> Array2<f32> {
        return Array2::random((input_size, output_size), Uniform::new(0., 1.0).unwrap());
    }
}

impl Module for FcLayer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        // store input for backprop computations
        self.last_input = Some(input.clone()); // cloning ...

        // (batch_size, input_size) X (input_size, output_size) = (batch_size, output_size)
        input.dot(&self.weights) + self.bias.clone()
    }

    fn backward(&mut self, next_layer_err: Array2<f32>) -> Array2<f32> {
        todo!()
    }
}

#[derive(Debug)]
struct ConvLayer {
    nb_channels: usize,
    height: usize,
    width: usize,
    kernels: Vec<Array4<f32>>, // (output_channels, input_channels, height, width)
    bias: Array1<f32>,         // (features)
}

impl ConvLayer {
    fn new(
        nb_channels: usize,
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
    ) -> ConvLayer {
        let mut kernels = Vec::new();
        for _ in 0..nb_channels {
            kernels.push(ConvLayer::init_conv_kernel(
                in_channels,
                out_channels,
                height,
                width,
            ));
        }

        ConvLayer {
            nb_channels,
            height,
            width,
            kernels,
            bias: ConvLayer::init_bias(out_channels), // TODO: check if this is really one
                                                      // bias per output channels
        }
    }

    fn init_conv_kernel(
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
    ) -> Array4<f32> {
        return Array4::random(
            (in_channels, out_channels, height, width),
            Uniform::new(0., 1.0).unwrap(),
        );
    }

    fn init_bias(output_size: usize) -> Array1<f32> {
        return Array1::random(output_size, Uniform::new(0., 1.0).unwrap());
    }
}

impl Module for ConvLayer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        todo!()
    }

    fn backward(&mut self, _post_grad: Array2<f32>) -> Array2<f32> {
        todo!()
    }
}

#[derive(Debug)]
enum Layer {
    FC(FcLayer),
    Conv(ConvLayer),
    ReLU,
    Softmax,
}

impl Module for Layer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        match self {
            Layer::FC(fc_layer) => fc_layer.forward(input),
            Layer::Conv(_) => todo!(),
            Layer::ReLU => todo!(),
            Layer::Softmax => todo!(),
        }
    }

    fn backward(&mut self, _post_grad: Array2<f32>) -> Array2<f32> {
        todo!()
    }
}

#[derive(Debug)]
struct NN {
    // layers: Vec<Box<dyn Module>>,
    layers: Vec<Layer>,
}

impl Module for NN {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        let mut x = input;
        for layer in &mut self.layers {
            x = layer.forward(x);
        }
        x
    }

    fn backward(&mut self, _post_grad: Array2<f32>) -> Array2<f32> {
        todo!()
    }
}

fn main() {
    let nn = NN {
        layers: vec![
            Layer::FC(FcLayer::new(1, 10)), // input will be the output lol, let's try to train the
            // 'identity' neural network!
            Layer::Softmax,
        ],
    };

    println!("My nn: {:?}", nn);
}
