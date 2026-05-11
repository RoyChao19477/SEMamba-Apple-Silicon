import gc
import glob
import os
import argparse
import json
import torch
import torch.nn.functional as F
import librosa
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import SEMamba
from models.pcs400 import cal_pcs
import soundfile as sf

from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed, 
    print_gpu_info, log_model_info, initialize_process_group,
)

h = None
device = None

def get_device():
    """Automatically detect and return the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _enhance_segment(noisy_wav, model, n_fft, hop_size, win_size, compress_factor):
    """Run the model on a single waveform segment."""
    segment_len = noisy_wav.size(-1)
    energy = torch.sum(noisy_wav ** 2.0)
    norm_factor = torch.sqrt(
        noisy_wav.new_tensor(segment_len, dtype=torch.float32) / (energy + 1e-9)
    )
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
    noisy_amp, noisy_pha, _ = mag_phase_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)
    amp_g, pha_g, _ = model(noisy_amp, noisy_pha)
    audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
    audio_g = (audio_g / norm_factor).squeeze(0)
    return audio_g[:segment_len]

def enhance_audio(noisy_wav, model, chunk_size, chunk_overlap, n_fft, hop_size, win_size, compress_factor):
    """Enhance full utterance, optionally chunking with overlap-add crossfades."""
    total_len = noisy_wav.size(-1)
    if not chunk_size or chunk_size <= 0 or total_len <= chunk_size:
        return _enhance_segment(noisy_wav, model, n_fft, hop_size, win_size, compress_factor)

    if chunk_overlap >= chunk_size:
        raise ValueError('Chunk overlap must be smaller than chunk size.')

    step = chunk_size - chunk_overlap
    output = torch.zeros(total_len, device=noisy_wav.device)
    weight = torch.zeros_like(output)
    window = torch.hamming_window(chunk_size, periodic=False, device=noisy_wav.device)

    start = 0
    while start < total_len:
        end = min(start + chunk_size, total_len)
        chunk = noisy_wav[start:end]
        valid_len = chunk.size(-1)
        if valid_len < chunk_size:
            chunk = F.pad(chunk, (0, chunk_size - valid_len))

        enhanced_chunk = _enhance_segment(chunk, model, n_fft, hop_size, win_size, compress_factor)
        window_slice = window
        if valid_len < chunk_size:
            window_slice = window.clone()
            window_slice[valid_len:] = 0.0

        overlap_len = min(valid_len, chunk_size)
        output[start:end] += enhanced_chunk[:overlap_len] * window_slice[:overlap_len]
        weight[start:end] += window_slice[:overlap_len]
        start += step

    weight = torch.where(weight > 0, weight, torch.ones_like(weight))
    return output / weight

def inference(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']
    default_chunk_size = cfg.get('training_cfg', {}).get('segment_size')

    model = SEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()

    chunk_size_samples = None
    chunk_overlap_samples = 0
    if args.chunk_size is not None and args.chunk_size > 0:
        chunk_size_samples = max(int(args.chunk_size * sampling_rate), hop_size)
    elif default_chunk_size:
        chunk_size_samples = int(default_chunk_size)

    if chunk_size_samples:
        chunk_overlap_seconds = max(args.chunk_overlap, 0.0)
        chunk_overlap_samples = int(chunk_overlap_seconds * sampling_rate)
        chunk_overlap_samples = min(chunk_overlap_samples, chunk_size_samples - 1)
        print(f'Chunked inference -> size: {chunk_size_samples} samples (~{chunk_size_samples / sampling_rate:.2f}s), overlap: {chunk_overlap_samples} samples.')
    else:
        print('Chunking disabled. Enhancing full utterances.')

    with torch.no_grad():
        # You can use data.json instead of input_folder with:
        # ---------------------------------------------------- #
        # with open("data/test_noisy.json", 'r') as json_file:
        #     test_files = json.load(json_file)
        # for i, fname in enumerate( test_files ): 
        #     folder_path = os.path.dirname(fname)
        #     fname = os.path.basename(fname)
        #     noisy_wav, _ = librosa.load(os.path.join( folder_path, fname ), sr=sampling_rate)
        #     noisy_wav = torch.FloatTensor(noisy_wav).to(device)
        # ---------------------------------------------------- #
        for i, fname in enumerate(os.listdir( args.input_folder )):
            print(fname, args.input_folder)
            noisy_wav, _ = librosa.load(os.path.join( args.input_folder, fname ), sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)

            audio_g = enhance_audio(
                noisy_wav,
                model,
                chunk_size_samples,
                chunk_overlap_samples,
                n_fft,
                hop_size,
                win_size,
                compress_factor,
            )

            output_file = os.path.join(args.output_folder, fname)

            if args.post_processing_PCS == True:
                audio_np = cal_pcs(audio_g.squeeze().cpu().numpy())
                sf.write(output_file, audio_np, sampling_rate, 'PCM_16')
            else:
                sf.write(output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')

            del noisy_wav, audio_g
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()


def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='/mnt/e/Corpora/noisy_vctk/noisy_testset_wav_16k/')
    parser.add_argument('--output_folder', default='results')
    parser.add_argument('--config', default='results')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--post_processing_PCS', type=str2bool, default=False)
    parser.add_argument('--device', type=str, default=None, help='Device to use: mps, cuda, or cpu. If not specified, auto-detects.')
    parser.add_argument('--chunk_size', type=float, default=None,
                        help='Chunk size in seconds. Defaults to training segment size; set <=0 to disable chunking.')
    parser.add_argument('--chunk_overlap', type=float, default=0.05,
                        help='Chunk overlap in seconds when chunking is enabled.')
    args = parser.parse_args()

    global device
    
    # Set device
    if args.device:
        device = torch.device(args.device)
        print(f'Using specified device: {device}')
    else:
        device = get_device()
        print(f'Auto-detected device: {device}')
    
    # Print device info
    if device.type == 'mps':
        print('Using Metal Performance Shaders (MPS) on Apple Silicon')
    elif device.type == 'cuda':
        print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
    else:
        print('Using CPU')

    inference(args, device)


if __name__ == '__main__':
    main()
