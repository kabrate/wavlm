from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector, WavLMModel,Wav2Vec2ForSequenceClassification
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf
import torch
import os
import pickle
import _pickle as cPickle

encoder_outputs = {}

# Define the hook function
def get_encoder_outputs(module, input, output):
    encoder_outputs[module] = output.detach()



# Load audio
def wavlm_frames(filename, type, save_path=''):
    name = os.path.basename(filename).replace('.flac', '.pckl')
    # name=os.path.basename(filename).replace('.wav', '.pckl')
    # y, sr = librosa.load(filename) #接近1秒
    waveform, sample_rate = sf.read(filename)  # 0.002~0.003
    # waveform, sample_rate = torchaudio.load('00001.wav')
    # y = y.reshape(y.shape[0], 1)
    # resampler = Resample(orig_freq=sample_rate, new_freq=16000)
    # waveform_resampled = resampler(waveform)
    # Initialize feature extractor and model
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')
    model = Wav2Vec2ForSequenceClassification.from_pretrained('microsoft/wavlm-large')
    #model = WavLMModel.from_pretrained('microsoft/wavlm-large')
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
    inputs['input_values'] = inputs['input_values'].squeeze(1)  # Correct approach
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # 将模型移到CUDA设备
    model = model.to(device)

    # 将输入数据移到CUDA设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 在不计算梯度的情况下执行前向传播
    with torch.no_grad():
        outputs = model(**inputs,output_hidden_states=True)
        #features = outputs.last_hidden_state
        logits = model(**inputs).logits
    # 如果需要，将特征数据移回CPU
    #features = features.cpu()
    save_file = os.path.join(save_path, name)
    f = open(save_file, 'wb')
    muldata = [features, type]
    cPickle.dump(muldata, f)  # pop noise数量
    f.close()
    print(save_file)

wavlm_frames('LA_D_5030185.flac', 1)




