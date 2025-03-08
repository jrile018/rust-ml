use yahoo_finance_api as yahoo;
use csv;
use std::error::Error;
use yahoo::time::{OffsetDateTime, Duration as TimeDuration};
use yahoo::time::format_description;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};
use tch::nn::Module;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // ============================
    // PART 1: Download Data and Write CSV
    // ============================
    let ticker = "AAPL";
    let provider = yahoo::YahooConnector::new()?;
    let end_dt = OffsetDateTime::now_utc();
    let start_dt = end_dt - TimeDuration::days(365);

    // Fetch daily quotes ("1d" interval)
    let resp = provider.get_quote_history_interval(ticker, start_dt, end_dt, "1d").await?;
    let quotes = resp.quotes()?;

    // Create CSV file
    let file_path = format!("{}_1yr_stock_data.csv", ticker);
    let mut wtr = csv::Writer::from_path(&file_path)?;
    wtr.write_record(&["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])?;

    let mut year_high = f64::MIN;
    let mut year_low = f64::MAX;
    let date_format = format_description::parse("[year]-[month]-[day]")?;

    for q in &quotes {
        if q.high > year_high { year_high = q.high; }
        if q.low < year_low { year_low = q.low; }
        let datetime = OffsetDateTime::from_unix_timestamp(q.timestamp as i64)?;
        let date_str = datetime.format(&date_format)?;
        wtr.write_record(&[
            &date_str, &q.open.to_string(), &q.high.to_string(), &q.low.to_string(),
            &q.close.to_string(), &q.adjclose.to_string(), &q.volume.to_string(),
        ])?;
    }
    wtr.flush()?;

    // Append summary rows
    wtr.write_record(&["", "", "", "", "", "", ""])?;
    wtr.write_record(&["52-Week High", &year_high.to_string(), "", "", "", "", ""])?;
    wtr.write_record(&["52-Week Low", &year_low.to_string(), "", "", "", "", ""])?;
    wtr.flush()?;

    println!("Wrote {} rows of data to {}", quotes.len(), file_path);
    println!("52-Week High: {}", year_high);
    println!("52-Week Low: {}", year_low);

    // ============================
    // PART 2: Load CSV, Create Sliding Window, Normalize, and Train Neural Network
    // ============================
    let window_size = 5;
    let (x_train, y_train) = load_data_sliding_window(&file_path, window_size)?;
    println!("Loaded {} training examples.", x_train.size()[0]);

    // Normalize features
    let x_mean = x_train.mean_dim(&[0], true, Kind::Float);
    let x_std = x_train.std_dim(&[0], false, false);
    let x_train_norm = (&x_train - &x_mean) / &x_std;

    let y_mean = y_train.mean(Kind::Float);
    let y_std = y_train.std(false);
    let y_train_norm = (y_train - &y_mean) / &y_std;

    // Define neural network
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let input_dim = x_train_norm.size()[1];
    let vs = nn::VarStore::new(device);
    let net = nn::seq()
        .add(nn::linear(&vs.root(), input_dim, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root(), 64, 32, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root(), 32, 16, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root(), 16, 1, Default::default()));

    let x_train_norm = x_train_norm.to_device(device);
    let y_train_norm = y_train_norm.to_device(device);

    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    let epochs = 1000;
    for epoch in 1..=epochs {
        let pred = net.forward(&x_train_norm);
        let loss = pred.mse_loss(&y_train_norm, tch::Reduction::Mean);
        opt.backward_step(&loss);
        if epoch % 100 == 0 {
            println!("Epoch: {:>4} Loss: {:.4}", epoch, f64::from(&loss));
        }
    }

    // Predict next day's opening price
    let last_feature = x_train_norm.get(-1);
    let pred_norm = net.forward(&last_feature.unsqueeze(0));
    let pred_denorm = &pred_norm * &y_std + &y_mean;
    println!("Predicted next day's opening price: {:.2}", f64::from(&pred_denorm));

    Ok(())
}

/// Loads CSV data and constructs training examples using a sliding window.
fn load_data_sliding_window<P: AsRef<std::path::Path>>(
    path: P,
    window_size: usize,
) -> Result<(Tensor, Tensor), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut records = Vec::new();
    
    for result in rdr.records() {
        let record = result?;
        let date_field = record.get(0).unwrap().trim();
        if !date_field.contains("-") { continue; }
        
        if record.get(1).unwrap().trim().is_empty() || record.get(2).unwrap().trim().is_empty() ||
           record.get(3).unwrap().trim().is_empty() || record.get(4).unwrap().trim().is_empty() ||
           record.get(6).unwrap().trim().is_empty() {
            continue;
        }
        
        let open: f32 = record.get(1).unwrap().parse()?;
        let high: f32 = record.get(2).unwrap().parse()?;
        let low: f32 = record.get(3).unwrap().parse()?;
        let close: f32 = record.get(4).unwrap().parse()?;
        let volume: f32 = record.get(6).unwrap().parse()?;
        records.push((open, high, low, close, volume));
    }

    let num_records = records.len();
    if num_records <= window_size {
        return Err("Not enough records for the specified window size".into());
    }

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    
    for i in 0..(num_records - window_size) {
        let window = &records[i..i + window_size];
        let moving_avg: f32 = window.iter().map(|&(_, _, _, close, _)| close).sum::<f32>() / window_size as f32;
        xs.push(window.iter().flat_map(|&(o, h, l, c, v)| vec![o, h, l, c, v]).chain(vec![moving_avg]).collect::<Vec<_>>());
        ys.push(records[i + window_size].0);
    }

    Ok((Tensor::of_slice2(&xs).to_kind(Kind::Float), Tensor::of_slice(&ys).to_kind(Kind::Float).unsqueeze(1)))
}
