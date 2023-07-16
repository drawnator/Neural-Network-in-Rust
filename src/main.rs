// basic implementation of a neural network in rust
//
// ~ Guilherme Silva Toledo
use minifb::{Key, Window, WindowOptions};

const NETWORK_SIZE:usize = 1;
const LEARNING_RATE:f32 = 0.005;
const TRAINING_SAMPLE:usize = 30;
const TRAINING_EPOCH:usize = 10000;

// 1 => re_lu,
// 2 => leaky,
// 3 => sigmoid,
// _ => binstep
const ACTIVATION_CHOICE:u8 = 2;

// 1 => linear,
// 2 => exponential,
// 3 => sine,
// _ => tangential
const TRAINING_CHOICE:u8 = 1;

const DEBUG:bool = true;
const WIDTH: usize = 640;
const HEIGHT: usize = 360;

fn main() {
    if DEBUG {println!("{:?}",create_training_set(TRAINING_CHOICE)(0.0,1.0));}

    let training = create_training_set(TRAINING_CHOICE)(0.0,5.0);
    // let validation = create_training_set(TRAINING_CHOICE)(0.0,3.0);
    let mut network = single_layer_nn();
    // for _epoch in 0..1000 {
    //     // println!("\n{}",calcuate_loss(training, network));
    //     train_round(&mut network, training);
    //     // println!("training {}", calculate_loss(training, network));
    //     // println!("validation {}", calculate_loss(validation, network));
    //     // io::stdin().read(&mut [0]).unwrap();
    // }
    draw_training(training,&mut network);
    // println!("training {}", calculate_loss(training, network));
    // println!("validation {}", calculate_loss(validation, network));
    if DEBUG {training_comparation(network,training);}
    draw_comparation(training,network)
}

// Debug functions
fn draw_training(training:[[f32;2];TRAINING_SAMPLE],network:&mut [[f32;4];NETWORK_SIZE]) {
    let min_border_x:f32 = -5.0 + training[0][0];
    let max_border_x:f32 = 5.0 + training[TRAINING_SAMPLE-1][0];
    let min_border_y:f32 = -5.0 + training[0][1];
    let max_border_y:f32 = 5.0 + training[TRAINING_SAMPLE-1][1];

    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut window = Window::new(
        "rust-minifb",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    let mut epoch = 0;

    while window.is_open() && !window.is_key_down(Key::Escape) && epoch < TRAINING_EPOCH {

        epoch += 1;
        println!("{}",calculate_loss(training, *network));
        // train_round(network, training);
        if window.is_key_down(Key::Space) {train_round(network, training);}
        if window.is_key_down(Key::A) {network[0][0] += LEARNING_RATE;}
        if window.is_key_down(Key::Z) {network[0][0] -= LEARNING_RATE;}
        if window.is_key_down(Key::S) {network[0][1] += LEARNING_RATE;}
        if window.is_key_down(Key::X) {network[0][1] -= LEARNING_RATE;}
        if window.is_key_down(Key::D) {network[0][2] += LEARNING_RATE;}
        if window.is_key_down(Key::C) {network[0][2] -= LEARNING_RATE;}
        if window.is_key_down(Key::F) {network[0][3] += LEARNING_RATE;}
        if window.is_key_down(Key::V) {network[0][3] -= LEARNING_RATE;}

        for i in buffer.iter_mut() {
            *i = 0; // write something more funny here!
        }
        for x in 0..WIDTH {
            let input = min_border_x + (max_border_x-min_border_x)*x as f32/WIDTH as f32;
            let y = h_window_transform(inference(input, *network), min_border_x, max_border_x);
            if is_valid_point(x,y) {
                buffer[y*WIDTH+x] = 0x00CCEEAA;
            }
        }
        for point in training{
            let x = l_window_transform(point[0], min_border_x, max_border_x);
            let y = h_window_transform(point[1], min_border_y, max_border_y);
            if is_valid_point(x,y) {
                buffer[y*WIDTH+x] = 0x00FF0000;
            }
        }

        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();
    }
}

fn draw_comparation(training:[[f32;2];TRAINING_SAMPLE],network:[[f32;4];NETWORK_SIZE]) {

    let min_border_x:f32 = -5.0 + training[0][0];
    let max_border_x:f32 = 5.0 + training[TRAINING_SAMPLE-1][0];
    let min_border_y:f32 = -5.0 + training[0][1];
    let max_border_y:f32 = 5.0 + training[TRAINING_SAMPLE-1][1];

    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut window = Window::new(
        "rust-minifb",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for i in buffer.iter_mut() {
            *i = 0; // write something more funny here!
        }
        for x in 0..WIDTH {
            let input = min_border_x + (max_border_x-min_border_x)*x as f32/WIDTH as f32;
            let y = l_window_transform(inference(input, network), min_border_x, max_border_x);
            if is_valid_point(x,y) {
                buffer[y*WIDTH+x] = 0x00CCEEAA;
            }
        }
        for point in training{
            let x = l_window_transform(point[0], min_border_x, max_border_x);
            let y = h_window_transform(point[1], min_border_y, max_border_y);
            if is_valid_point(x,y) {
                buffer[y*WIDTH+x] = 0x00FF0000;
            }
        }

        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();
    }

}

fn l_window_transform(x:f32, min_border_x:f32, max_border_x:f32) -> usize {
    ((x-min_border_x)/(max_border_x-min_border_x)*WIDTH as f32) as usize
}

fn h_window_transform(y:f32, min_border_y:f32, max_border_y:f32) -> usize {
    ((y-min_border_y)/(max_border_y-min_border_y)*HEIGHT as f32) as usize
}

fn is_valid_point(x:usize, y:usize) -> bool {
    x < WIDTH && y < HEIGHT && x > 0 && y > 0
}

// fn basic_comparation(network:[[f32;4];2]) {
//     let mut output: [[f32;2];100] = [[0.0;2];100];
//     for x in 0..100 {
//         let y = inference(x as f32/100.0, network);
//         output[x][0] = x as f32/100.0;
//         output[x][1] = y;
//     }
//     println!("{:?}", output);
// }

fn training_comparation(network:[[f32;4];NETWORK_SIZE],training:[[f32;2];TRAINING_SAMPLE]){
    let mut output: [[f32;3];TRAINING_SAMPLE] = [[0.0;3];TRAINING_SAMPLE];
    for training_index in 0..30 {
        let x = training[training_index][0];
        let y = inference(x, network);
        let z = training[training_index][1];
        output[training_index][0] = x;
        output[training_index][1] = y;
        output[training_index][2] = z;
    }
    println!("{:?}", output);
}

// neural network functions
fn single_layer_nn() -> [[f32;4];NETWORK_SIZE]{
    [[0.0;4];NETWORK_SIZE]
}

fn inference(input:f32,network:[[f32;4];NETWORK_SIZE]) -> f32{
    let mut output = 0.0;
    for node in network {
        output += activation(ACTIVATION_CHOICE)
        (input.mul_add(node[0], node[1])).mul_add(node[2], node[3]);
    }
    output
}

fn calculate_loss(training:[[f32;2];TRAINING_SAMPLE],network:[[f32;4];NETWORK_SIZE]) -> f32{
    let mut loss = 0.0;
    for supervision in training {
        loss += (inference(supervision[0], network) - supervision[1]).abs()
    }
    loss.abs()
}

fn train(node:usize, wb:usize,network:&mut [[f32;4];NETWORK_SIZE],training:[[f32;2];TRAINING_SAMPLE]) {
    let default_loss = calculate_loss(training, *network);
    network[node][wb] += LEARNING_RATE;
    if calculate_loss(training, *network) > default_loss {
        network[node][wb] -= 2.0*LEARNING_RATE;
    }
}

fn train_round(network:&mut [[f32;4];NETWORK_SIZE], training:[[f32;2];TRAINING_SAMPLE]){
    for node in 0..NETWORK_SIZE {
        for wb in 0..4 {
            train(node, wb, network, training)
        }
    }
}

// activation functions
fn activation(choice:u8) -> fn(f32) -> f32{
    match choice {
        1 => re_lu,
        2 => leaky,
        3 => sigmoid,
        _ => binstep
    }
}

fn re_lu(number:f32) -> f32 {
    if number < 0.0 {0.0} else {number}
}

fn sigmoid(number:f32) -> f32 {
    1.0/(1.0+(-number).exp())
}

fn leaky(number:f32) -> f32 {
    if number < 0.0 {0.01*number} else {number}
}

fn binstep(number:f32) -> f32 {
    if number < 0.0 {0.0} else {1.0}
}

// training functions 
fn create_training_set(choice:u8) -> fn(f32,f32) -> [[f32;2];TRAINING_SAMPLE] {
    match choice {
        1 => linear,
        2 => exponential,
        3 => sine,
        _ => tangential
    }
}

fn linear(start:f32,end:f32) -> [[f32;2];TRAINING_SAMPLE] {
    let mut linear_set:[[f32;2];TRAINING_SAMPLE] = [[0.0;2];TRAINING_SAMPLE];
    for i in 0..TRAINING_SAMPLE {
        let x = linear_interpolation(start, end, i);
        linear_set[i] = [x,x];
    }
    linear_set
}

fn exponential(start:f32,end:f32) -> [[f32;2];TRAINING_SAMPLE] {
    let mut exponential_set:[[f32;2];TRAINING_SAMPLE] = [[0.0;2];TRAINING_SAMPLE];
    for i in 0..TRAINING_SAMPLE {
        let x = linear_interpolation(start, end, i);
        exponential_set[i] = [x,x.exp()];
    }
    exponential_set
}

fn sine(start:f32,end:f32) -> [[f32;2];TRAINING_SAMPLE] {
    let mut sine_set:[[f32;2];TRAINING_SAMPLE] = [[0.0;2];TRAINING_SAMPLE];
    for i in 0..TRAINING_SAMPLE {
        let x = linear_interpolation(start, end, i);
        sine_set[i] = [x,x.sin()];
    }
    sine_set
}

fn tangential(start:f32,end:f32) -> [[f32;2];TRAINING_SAMPLE] {
    let mut tangential_set:[[f32;2];TRAINING_SAMPLE] = [[0.0;2];TRAINING_SAMPLE];
    for i in 0..TRAINING_SAMPLE {
        let x = linear_interpolation(start, end, i);
        tangential_set[i] = [x,x.tan()];
    }
    tangential_set
}

//TODO test other implementations
fn linear_interpolation(start:f32,end:f32,i:usize) -> f32 {
    (i as f32).mul_add((end-start)/TRAINING_SAMPLE as f32, start)
}