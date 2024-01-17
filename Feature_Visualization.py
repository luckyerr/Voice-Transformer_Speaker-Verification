import matplotlib.pyplot as plt
import os
import torch
import torchaudio

output_images_folder = "path1"  # Directory to save images
data_path_list ="path2"         # Directory to input videos

all_frames = 0
all_mean_stat = torch.zeros(112)  
all_var_stat = torch.zeros(112)





for root, dirs, files in os.walk(data_path_list):
    rel_path = os.path.relpath(root, data_path_list)
    output_folder = os.path.join(output_images_folder, rel_path)

    for wav_file in files:
        if wav_file.endswith('.wav'):
            wav_path = os.path.join(root, wav_file)
            print(wav_path)

            waveform, sample_rate = torchaudio.load(wav_path)
            feature = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=111, dither=0, energy_floor=0,
                                                         low_freq=20, high_freq=0, preemphasis_coefficient=0.97,
                                                         use_energy=True, raw_energy=True, remove_dc_offset=True,
                                                         sample_frequency=16000, window_type='povey')
            #feature =torchaudio.compliance.kaldi.mfcc(waveform)
            xs = feature

            plt.imshow(xs.detach().numpy(), cmap='viridis', origin='lower', aspect='auto')
            plt.axis("off")


            output_filename = os.path.splitext(os.path.basename(wav_path))[0] + '.png'
            os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
            output_path = os.path.join(output_folder, output_filename)
            plt.savefig(output_path,bbox_inches="tight", pad_inches=0)
            plt.show() 
