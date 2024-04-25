from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector, WavLMModel,Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from transformers import TrainingArguments
from transformers import Trainer
from dataset1 import ASVspoof2019Dataset,collate_fn
import os
asv2019_path='D:/dataset/LA/LA/'

processor =Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')
model = Wav2Vec2ForSequenceClassification.from_pretrained('microsoft/wavlm-large')
train_dataset = ASVspoof2019Dataset(os.path.join(asv2019_path,'ASVspoof2019_LA_train/flac/') , os.path.join(asv2019_path,'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'))
eval_dataset = ASVspoof2019Dataset('D:/dataset/LA/LA/ASVspoof2019_LA_train/flac/',os.path.join(asv2019_path,'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'))

training_args = TrainingArguments(
    output_dir="./models/anti_spoofing",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir='./logs',
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,  # Pass your custom collate function
)

trainer.train()
trainer.evaluate()
model.save_pretrained("./models/wavlm-finetuned")
processor.save_pretrained("./models/wavlm-finetuned")


