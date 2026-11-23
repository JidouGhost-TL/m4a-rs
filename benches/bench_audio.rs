use std::hint::black_box;
use std::time::Instant;

use m4a::{AccuracySynth, M4aSampleView, M4aSoundChannel, StudioSynth, WaveDataHeader};

fn main() {
    println!("=== m4a-rs Audio Performance Benchmarks ===");

    let pcm = vec![48u8; 4096];
    let header = WaveDataHeader {
        wave_type: 0,
        stat: 0,
        flags: 0x40,
        base_freq_q10: 13_700_096,
        loop_start: 0,
        size: 4092,
    };
    let sample = M4aSampleView::from_parts(header, &pcm).unwrap();
    let mut out_r = vec![0i8; 224];
    let mut out_l = vec![0i8; 224];
    let iters = 100_000;

    // 1. AccuracySynth fixed 1:1 pitch
    let mut ch_fixed = M4aSoundChannel::new(0x08, 220, 110, 255, 255, 255, 0, 13_379, 0);
    for _ in 0..1000 {
        out_r.fill(0);
        out_l.fill(0);
        AccuracySynth::mix_channel_frame(&sample, &mut ch_fixed, 13_379, 15, &mut out_r, &mut out_l);
    }
    let start = Instant::now();
    for _ in 0..iters {
        out_r.fill(0);
        out_l.fill(0);
        AccuracySynth::mix_channel_frame(&sample, &mut ch_fixed, 13_379, 15, &mut out_r, &mut out_l);
        black_box((&out_r, &out_l));
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_nanos() as f64 / iters as f64;
    let msps = ((iters * 224) as f64 / elapsed.as_secs_f64()) / 1_000_000.0;
    println!("AccuracySynth fixed 1:1 (224 frames):        {ns:8.2} ns/iter ({msps:7.2} MSamples/sec)");

    // 2. AccuracySynth resampled Q23 linear interpolation
    let mut ch_resamp = M4aSoundChannel::new(0x00, 200, 140, 255, 255, 255, 0, 18_500, 0);
    for _ in 0..1000 {
        out_r.fill(0);
        out_l.fill(0);
        AccuracySynth::mix_channel_frame(&sample, &mut ch_resamp, 13_379, 15, &mut out_r, &mut out_l);
    }
    let start = Instant::now();
    for _ in 0..iters {
        out_r.fill(0);
        out_l.fill(0);
        AccuracySynth::mix_channel_frame(&sample, &mut ch_resamp, 13_379, 15, &mut out_r, &mut out_l);
        black_box((&out_r, &out_l));
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_nanos() as f64 / iters as f64;
    let msps = ((iters * 224) as f64 / elapsed.as_secs_f64()) / 1_000_000.0;
    println!("AccuracySynth resampled Q23 (224 frames):    {ns:8.2} ns/iter ({msps:7.2} MSamples/sec)");

    // 3. AccuracySynth multi-voice (3 channels: 1 fixed + 2 resampled)
    let mut ch0 = M4aSoundChannel::new(0x08, 128, 64, 255, 255, 255, 0, 13_379, 0);
    let mut ch1 = M4aSoundChannel::new(0x00, 64, 128, 255, 255, 255, 0, 16_500, 100);
    let mut ch2 = M4aSoundChannel::new(0x00, 100, 100, 255, 255, 255, 0, 9_800, 200);
    for _ in 0..1000 {
        out_r.fill(0);
        out_l.fill(0);
        AccuracySynth::mix_channel_frame(&sample, &mut ch0, 13_379, 15, &mut out_r, &mut out_l);
        AccuracySynth::mix_channel_frame(&sample, &mut ch1, 13_379, 15, &mut out_r, &mut out_l);
        AccuracySynth::mix_channel_frame(&sample, &mut ch2, 13_379, 15, &mut out_r, &mut out_l);
    }
    let start = Instant::now();
    for _ in 0..iters {
        out_r.fill(0);
        out_l.fill(0);
        AccuracySynth::mix_channel_frame(&sample, &mut ch0, 13_379, 15, &mut out_r, &mut out_l);
        AccuracySynth::mix_channel_frame(&sample, &mut ch1, 13_379, 15, &mut out_r, &mut out_l);
        AccuracySynth::mix_channel_frame(&sample, &mut ch2, 13_379, 15, &mut out_r, &mut out_l);
        black_box((&out_r, &out_l));
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_nanos() as f64 / iters as f64;
    let msps = ((iters * 224) as f64 / elapsed.as_secs_f64()) / 1_000_000.0;
    println!("AccuracySynth 3-voice stereo (224 frames):   {ns:8.2} ns/iter ({msps:7.2} MSamples/sec)");

    // 4. AccuracySynth reverb feedback pass
    let mut next_r = vec![24i8; 224];
    let mut next_l = vec![-16i8; 224];
    for _ in 0..1000 {
        AccuracySynth::reverb_pass(&mut out_r, &mut out_l, &next_r, &next_l, 64);
    }
    let start = Instant::now();
    for _ in 0..iters {
        AccuracySynth::reverb_pass(&mut out_r, &mut out_l, &next_r, &next_l, 64);
        black_box((&out_r, &out_l, &mut next_r, &mut next_l));
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_nanos() as f64 / iters as f64;
    let msps = ((iters * 224) as f64 / elapsed.as_secs_f64()) / 1_000_000.0;
    println!("AccuracySynth reverb pass (224 frames):      {ns:8.2} ns/iter ({msps:7.2} MSamples/sec)");

    // 5. StudioSynth 48 kHz Lanczos f32
    let mut f_l = vec![0.0f32; 800];
    let mut f_r = vec![0.0f32; 800];
    let mut phase = 0.0f64;
    let studio_iters = 5_000;
    for _ in 0..100 {
        f_l.fill(0.0);
        f_r.fill(0.0);
        StudioSynth::mix_voice_f32(&sample, &mut phase, 18_500.0, 48_000.0, 0.8, 0.5, &mut f_l, &mut f_r);
    }
    let start = Instant::now();
    for _ in 0..studio_iters {
        f_l.fill(0.0);
        f_r.fill(0.0);
        StudioSynth::mix_voice_f32(&sample, &mut phase, 18_500.0, 48_000.0, 0.8, 0.5, &mut f_l, &mut f_r);
        black_box((&f_l, &f_r));
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_nanos() as f64 / studio_iters as f64;
    let msps = ((studio_iters * 800) as f64 / elapsed.as_secs_f64()) / 1_000_000.0;
    println!("StudioSynth Lanczos 48kHz (800 frames):      {ns:8.2} ns/iter ({msps:7.2} MSamples/sec)");
}
