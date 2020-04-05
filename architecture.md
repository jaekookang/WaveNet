# WaveNet Architecture Note

This document summarizes the WaveNet model architecture in this repository (written by CODEJIN) for personal understanding and later use. 

2020-04-01 jaekoo

## 1. Generate Pattern (`Pattern_Generator.py`)
- Load wav file lists (eg. `def LJ_Info_Load`)
- Generate {Singal, Mel, Speaker_ID, Dataset_meta} using ThreadPoolExecutor
- Generate meta information (for me, save train-specific hparams separately)

## 2. Feeder
- Feeder for training: 
	- Output (inserted in the dict `self.pattern_Queue`):
		- 'audios': sig_Pattern (10,769) (eg. 10 means 10 audio files for test)
		- 'mels': mel_Pattern (10,7,80)
		- 'speakers': speaker_Pattern (len=10)
- Feeder for inference: consider `batch_size` (eg. 16), `split_mel_window` (eg. 7), `overlap_Window=1` (for mel; mel itself is also overlapped and extracted)
	- Output: 
		- wav_List: eg. len=2
			- [0]: (57312,)
			- [1]: (117808,)
		- pattern_Dict_List: len=8
			- [0]
				- 'audios': (16,1792) => all zeros? because it is not teacher forcing
				- 'mels': (16,11,80)
				- 'speakers': (16,)
			- ...
		- split_Mel_Index_List: len=115 (fm=11,fs=5)

## 3. Modules
- class WaveNet (`tfk.Model`)
	- Training (inputs: x, local_Conditions (mel), global_Conditions (spkr))
		- x = layer_Dict['First'] <-- `Incremental_Conv1D_Causal_WN`
		- local_Conditions = layer_Dict['Local_Condition_Upsample']
		- global_Conditions = layer_Dict['Global_Condition_Embedding']
		- (for loop) x = layer_Dict['ResConvGLU\_{}\_{}'.format(block_Index, stack_Index)]
			- inputs=[x, local_Conditions, global_Conditions]
		- logits = layer_Dict['Last']
	- Inference (inputs: local_Conditions, global_Conditions)
		- Initialize layer_Dict['first'], layer_Dict['ResConvGLU\_{}\_{}'.format(block_Index, stack_Index)]
		- local_Conditions = layer_Dict['Local_Condition_Upsample']
		- global_Conditions = layer_Dict['Global_Condition_Embedding']
		- initial_Samples
		- (while loop) Similar to Training, but x is initialized internally
	- For example,
		- x = new_WaveNet(inputs=[x, locals, globals], training= True)
			- x: (3, 768) float32
			- locals: (3, 7, 80) float32, Mel
			- globals: np.array([1,]) int32, Spkr
		- (output) x: len=2 list
			- [0]: (3, 768, 30) <- 3 examples, 768 samples, 30 MoL dim
			- [1]: (3, 1)       <- zeros


- class UpsampleNet (`tfk.Model`)
	- It is `tfkl.UpSampling2D` with weight normalization

- class ResConvGLU (`tfkl.Layer`)
	- `Incremental_Conv1D_Causal_WN`
	- `tfkl.Conv1D`
		- layer_Dict['Out']
		- layer_Dict['Skip']
		- layer_Dict['Local_Condition_Conv1D']
		- layer_Dict['Global_Condition_Conv1D']
	- weight normalization

- class Weight_Norm_Wrapper 
	- Replace it with `import tensorflow_addons as tfa; tfa.layers.WeightNormalization`

- class Incremental_Conv1D_Causal_WN
	- WaveNet architecture
	- Training: `def usual(self, inputs)`
		- zero padding
		- `tf.nn.conv1d`
		- output: logits, zeros
	- Inference: `def incremental(self, inputs)`
		- `tf.nn.conv1d`
		- output: zeros, samples

- class Loss
	- `Discretized_Mix_Logistic_Loss(labels= labels, logits= logits)`

- class ExponentialDecay
	- `tfk.optimizers.schedules.ExponentialDecay`


## 4. Model
- class WaveNet
	- Wrapper for `Modules.WaveNet`
	- Training: `def Train(self)`
	- Inference: `def Inference(self)`

## 5. MoL
- `Discretized_Mix_Logistic_Loss(labels,logits,classes= 65536,log_scale_min= None)`



