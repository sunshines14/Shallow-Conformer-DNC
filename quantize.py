import torch
import numpy as np
import os

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    
num_classes = 10

sampling_rate = 44100
num_audio_channels = 1 
sample_duration = 10
num_freq_bins = 128
num_fft_points = 2048
hop_length = int(num_fft_points/2)
num_time_bins = int(np.ceil(sample_duration*sampling_rate / hop_length))

encoder_dim = 32
num_layers = 2
num_attention_heads = 2
feed_forward_expansion_factor = 4
conv_expansion_factor = 2 
dropout_p = 0.1
conv_kernel_size = 7
    
dnc_input_size = 32
dnc_hidden_size = 16
dnc_rnn_type = 'gru'
dnc_num_layers = 2
dnc_nr_cells = 16
dnc_cell_size = 16
dnc_read_heads = 2
dnc_gpu_id = 0
    
split_apply = False
extractor_apply = False
    
if torch.cuda.is_available():
    device = 'cuda'
    pin_memory = True
else:
    device = 'cpu'
    pin_memory = False     
    
from networks import conformer
model = conformer(num_classes = num_classes, 
                  input_dim = num_freq_bins, 
                  encoder_dim = encoder_dim, 
                  num_layers = num_layers, 
                  num_attention_heads = num_attention_heads,
                  feed_forward_expansion_factor = feed_forward_expansion_factor,
                  conv_expansion_factor = conv_expansion_factor,
                  dropout_p = dropout_p,
                  conv_kernel_size = conv_kernel_size,
                  dnc_input_size = dnc_input_size,
                  dnc_hidden_size = dnc_hidden_size,
                  dnc_rnn_type = dnc_rnn_type,
                  dnc_num_layers = dnc_num_layers,
                  dnc_nr_cells = dnc_nr_cells,
                  dnc_cell_size = dnc_cell_size,
                  dnc_read_heads = dnc_read_heads,
                  dnc_gpu_id = dnc_gpu_id,
                  device = device,
                  split_apply = split_apply,
                  extractor_apply = extractor_apply
                 ).to(device)

checkpoint_path = '/data/soonshin/asc/dcase-2021-pytorch/save/2103110604_1a_dev_300/epoch_123.pth'
model.load_state_dict(torch.load(checkpoint_path))
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

print_size_of_model(model)
print_size_of_model(quantized_model)