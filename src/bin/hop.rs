use engram::{ItemMemory, encode_image_bag};
use hopfield::hopfield::Hopfield;
use hopfield::state::State;
use hypervector::binary_hdv::BinaryHDV;
use mnist::error::MnistError;
use mnist::{self, Mnist};
use std::io::Write;

const N: usize = 100;
const IDIM: usize = N * usize::BITS as usize + 2 * 8;

fn decode_label(lab: u16) -> Option<u8> {
    // Keep only bits 0..=9 (10 valid digits for MNIST).
    let masked = lab & 0x03FF; // 0000_0011_1111_1111 = lower 10 bits

    // If no bit set or multiple bits set → invalid.
    if masked == 0 || masked.count_ones() != 1 {
        return None;
    }

    Some(masked.trailing_zeros() as u8)
}

fn is_power_of_two(n: u16) -> bool {
    let n = n & 0b11_1111_1111;
    n != 0 && n & (n - 1) == 0
}

// start with blank label and let the network reconstruct it as it settles in to an energy minimum
fn classify(net: &Hopfield<IDIM>, x: &mut State<IDIM>) -> Vec<u8> {
    //let mut g0 = net.goodness(x);
    let mut one_hot: u16 = 0;
    for it in 0..4 {
        net.step(x);
        //let g1 = net.goodness(x);
        let bytes: &[u8] = &x.bits[0].to_le_bytes();
        one_hot = u16::from_le_bytes([bytes[0], bytes[1]]);
        if is_power_of_two(one_hot) && it > 0 {
            break;
        }
        //if g1 == g0 {
        //    break;
        //}
        //g0 = g1
    }
    (0..10).filter(|i| one_hot & (1 << i) != 0).collect()
}

fn image_to_state<const N: usize>(img_hdv: &BinaryHDV<N>, digit: u8) -> State<IDIM> {
    let one_hot: u16 = 1 << digit;

    // Lazy iterators over label and image data
    let label_byte_iter = one_hot.to_le_bytes().into_iter();
    let image_byte_iter = img_hdv.data.iter().flat_map(|word| word.to_le_bytes());
    let combined_iter = label_byte_iter.chain(image_byte_iter);
    State::from_byte_iter(combined_iter)
}

fn main() -> Result<(), MnistError> {
    let imem = ItemMemory::<N>::new();
    let data = Mnist::load("MNIST")?;

    let net = if true {
        let mut net = Hopfield::<IDIM>::new();
        let epochs = 1;
        for j in 0..epochs {
            for (i, (im, &digit)) in data
                .train_images
                .iter()
                .zip(data.train_labels.iter())
                .enumerate()
            {
                let img_hdv = encode_image_bag(im.as_u8_array(), &imem);
                let state = image_to_state(&img_hdv, digit);
                net.perceptron_conv_procedure(&state);
                if i % 100 == 0 {
                    print!("Epoch {j}, Digit {i}/{}\r", data.train_images.len());
                    let _ = std::io::stdout().flush();
                }
            }
            let fname = format!("hop{j}.json");
            net.save_json(&fname).expect("failed to save");
        }
        net
    } else {
        Hopfield::<IDIM>::load_json("hop0.json")?
    };

    let (mut correct, mut n_ambiguous, mut no_result, mut error) = (0, 0, 0, 0);
    for (im, &digit) in data.test_images.iter().zip(data.test_labels.iter()) {
        let img_hdv = encode_image_bag(im.as_u8_array(), &imem);
        let mut state = image_to_state(&img_hdv, digit);
        let vam: Vec<u8> = classify(&net, &mut state);
        match vam.as_slice() {
            [predicted] if digit == *predicted => correct += 1,
            [_predicted] => {
                error += 1;
                println!("classify {digit}: {vam:?}\n{im}");
            }
            [] => {
                no_result += 1;
                println!("classify {digit}: {vam:?}\n{im}");
            }
            _ => {
                n_ambiguous += 1;
                println!("classify {digit}: {vam:?}\n{im}");
            }
        }
    }

    let total = data.test_images.len();
    let unambiguous = total - n_ambiguous - no_result;

    let pct = |n| 100.0 * n as f64 / total as f64;
    println!("ambiguous {n_ambiguous}/{total} = {:.2}%", pct(n_ambiguous));
    println!("no result {no_result}/{total} = {:.2}%", pct(no_result));
    println!("correct/total {correct}/{total} = {:.2}%", pct(correct));

    if unambiguous > 0 {
        let pct_u = |n| 100.0 * n as f64 / unambiguous as f64;
        println!(
            "correct/unambiguous {correct}/{unambiguous} = {:.2}%",
            pct_u(correct)
        );
        println!(
            "errors/unambiguous {error}/{unambiguous} = {:.2}%",
            pct_u(error)
        );
    }

    Ok(())
}
