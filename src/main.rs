const NETWORK_SIZE:usize = 2;
const TRAINING_SAMPLE:usize = 30;
const ACTIVATION_CHOICE:u8 = 1;
const TRAINING_CHOICE:u8 = 1;
const LEARNING_RATE:f32 = 0.01;
const DEBUG:bool = false;
fn main() {
    if DEBUG {println!("{:?}",create_training_set(TRAINING_CHOICE)(0.0,1.0));}

    let training = create_training_set(TRAINING_CHOICE)(0.0,1.0);
    let mut network = single_layer_nn();

    println!("{}",calcuate_loss(training, network));
    train_round(&mut network, training);
    println!("{}", calcuate_loss(training, network))
}

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

fn calcuate_loss(training:[[f32;2];TRAINING_SAMPLE],network:[[f32;4];NETWORK_SIZE]) -> f32{
    let mut loss = 0.0;
    for supervision in training {
        loss = inference(supervision[0], network) - supervision[1]
    }
    loss.abs()
}

fn train(node:usize, wb:usize,network:&mut [[f32;4];NETWORK_SIZE],training:[[f32;2];TRAINING_SAMPLE]) {
    let default_loss = calcuate_loss(training, *network);
    network[node][wb] += LEARNING_RATE;
    if calcuate_loss(training, *network) > default_loss {
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