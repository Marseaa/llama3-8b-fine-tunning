Fine tunning do modelo llama3 8b utilizando unsloth + inferencia simples

Fine tunning:

2024-06-12 14:46:48 ==========
2024-06-12 14:46:48 == CUDA ==
2024-06-12 14:46:48 ==========
2024-06-12 14:46:48 
2024-06-12 14:46:48 CUDA Version 11.8.0
2024-06-12 14:46:48 
2024-06-12 14:46:48 Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2024-06-12 14:46:48 
2024-06-12 14:46:48 This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2024-06-12 14:46:48 By pulling and using the container, you accept the terms and conditions of this license:
2024-06-12 14:46:48 https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2024-06-12 14:46:48 
2024-06-12 14:46:48 A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2024-06-12 14:46:48 
2024-06-12 14:46:51 🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
2024-06-12 14:46:52 ==((====))==  Unsloth: Fast Llama patching release 2024.5
2024-06-12 14:46:52    \\   /|    GPU: NVIDIA GeForce RTX 4090 Laptop GPU. Max memory: 15.992 GB. Platform = Linux.
2024-06-12 14:46:52 O^O/ \_/ \    Pytorch: 2.3.0+cu121. CUDA = 8.9. CUDA Toolkit = 12.1.
2024-06-12 14:46:52 \        /    Bfloat16 = TRUE. Xformers = 0.0.26.post1. FA = False.
2024-06-12 14:46:52  "-____-"     Free Apache license: http://github.com/unslothai/unsloth
2024-06-12 14:46:52 Unsloth: unsloth/llama-3-8b-Instruct-bnb-4bit can only handle sequence lengths of at most 8192.
2024-06-12 14:46:52 But with kaiokendev's RoPE scaling of 3.0, it can be magically be extended to 24576!
2024-06-12 14:50:42 Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
2024-06-12 14:50:45 Unsloth 2024.5 patched 32 layers with 32 QKV layers, 32 O layers and 32 MLP layers.
2024-06-12 14:50:49 /usr/local/lib/python3.10/dist-packages/transformers/training_args.py:1965: FutureWarning: `--push_to_hub_token` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--hub_token` instead.
2024-06-12 14:50:49   warnings.warn(
2024-06-12 14:50:49 /usr/local/lib/python3.10/dist-packages/trl/trainer/sft_trainer.py:269: UserWarning: You passed a `max_seq_length` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`.
2024-06-12 14:50:49   warnings.warn(
2024-06-12 14:50:49 /usr/local/lib/python3.10/dist-packages/trl/trainer/sft_trainer.py:283: UserWarning: You passed a `dataset_num_proc` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`.
2024-06-12 14:50:49   warnings.warn(
2024-06-12 14:50:49 /usr/local/lib/python3.10/dist-packages/trl/trainer/sft_trainer.py:307: UserWarning: You passed a `dataset_text_field` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`.
2024-06-12 14:50:49   warnings.warn(
2024-06-12 14:51:00 
Map (num_proc=2): 100%|█████████████████████████████████| 51760/51760 [00:10<00:00, 5007.01 examples/s]
2024-06-12 14:51:00 max_steps is given, it will override any value given in num_train_epochs
2024-06-12 14:51:00 
2024-06-12 14:51:00 Tudo certo, treinando dados...
2024-06-12 14:51:00 ==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = 1
2024-06-12 14:51:00    \\   /|    Num examples = 51,760 | Num Epochs = 1
2024-06-12 14:51:00 O^O/ \_/ \    Batch size per device = 2 | Gradient Accumulation steps = 4
2024-06-12 14:51:00 \        /    Total batch size = 8 | Total steps = 300
2024-06-12 14:51:00  "-____-"     Number of trainable parameters = 167,772,160
2024-06-12 15:08:39 
100%|████████████████████████████████████████████████████████████████| 300/300 [17:38<00:00,  3.31s/it]
100%|████████████████████████████████████████████████████████████████| 300/300 [17:38<00:00,  3.53s/it]
2024-06-12 15:08:39 Fim do treinamento de dados. Salvando no huggingface...
2024-06-12 15:08:39 
2024-06-12 15:08:40 Unsloth: Merging 4bit and LoRA weights to 16bit...
2024-06-12 15:08:40 Unsloth: Will use up to 17.6 out of 31.17 RAM for saving.
2024-06-12 15:08:41
2024-06-12 15:10:36 Unsloth: Saving tokenizer... Done.
2024-06-12 15:10:36 Unsloth: Saving model... This might take 5 minutes for Llama-7b...
2024-06-12 15:16:07 Done.
2024-06-12 15:16:12 Unsloth: You are pushing to hub, but you passed your HF username = Marseaa.
2024-06-12 15:16:12 We shall truncate Marseaa/llama3-8b-fine-tunned to llama3-8b-fine-tunned
2024-06-12 15:16:12 Unsloth: Merging 4bit and LoRA weights to 16bit...
2024-06-12 15:16:12 Unsloth: Will use up to 17.59 out of 31.17 RAM for saving.
2024-06-12 15:16:58 
100%|██████████████████████████████████████████████████████████████████| 32/32 [00:45<00:00,  1.41s/it]
2024-06-12 15:17:01 Unsloth: Saving tokenizer... Done.
2024-06-12 15:17:01 Unsloth: Saving model... This might take 5 minutes for Llama-7b...
model-00003-of-00004.safetensors: 100%|███████████████████████████| 4.92G/4.92G [22:49<00:00, 3.59MB/s]
2024-06-12 15:44:46 .safetensors: 100%|██████████████████████████▉| 4.91G/4.92G [22:46<00:00, 12.6MB/s]
Upload 4 LFS files:  75%|███████████████████████████████████▎           | 3/4 [22:49<05:15, 315.71s/it]
Upload 4 LFS files: 100%|███████████████████████████████████████████████| 4/4 [22:49<00:00, 342.36s/it]
2024-06-12 15:44:47 Done.
2024-06-12 15:44:47 Saved merged model to https://huggingface.co/Marseaa/llama3-8b-fine-tunned
2024-06-12 15:44:53 
2024-06-12 15:44:53 Fim do salvamento.



Inferencia:

2024-06-12 16:51:33 
2024-06-12 16:51:33 ==========
2024-06-12 16:51:33 == CUDA ==
2024-06-12 16:51:33 ==========
2024-06-12 16:51:33 
2024-06-12 16:51:33 CUDA Version 12.2.2
2024-06-12 16:51:33 
2024-06-12 16:51:33 Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2024-06-12 16:51:33 
2024-06-12 16:51:33 This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2024-06-12 16:51:33 By pulling and using the container, you accept the terms and conditions of this license:
2024-06-12 16:51:33 https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2024-06-12 16:51:33 
2024-06-12 16:51:33 A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2024-06-12 16:51:33 
2024-06-12 16:51:34 Cuda disponível, rodando na gpu...
2024-06-12 16:51:34 
2024-06-12 16:51:34 
2024-06-12 16:51:34 ----GERADOR DE TEXTO----
2024-06-12 16:51:34 
2024-06-12 16:51:34 
2024-06-12 16:52:35 Insira o texto inicial: Can you explain to me something marine biology related?
Loading checkpoint shards: 100%|█████████████████████████████████████████| 4/4 [03:06<00:00, 39.61s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████| 4/4 [03:06<00:00, 46.74s/
2024-06-12 16:55:46 Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
2024-06-12 16:55:46 Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
2024-06-12 16:55:50 
2024-06-12 16:55:50 
2024-06-12 16:55:50  Texto gerado: 
2024-06-12 16:55:50 
2024-06-12 16:55:50 <|begin_of_text|>Can you explain to me something marine biology related? I'm interested in learning more about the ocean and its inhabitants.
2024-06-12 16:55:50 
2024-06-12 16:55:50 ### Response:
2024-06-12 16:55:50 I'd be happy to explain something related to marine biology. What specific topic would you like to learn about? Are you interested in learning about a particular type of marine animal, such as coral, fish, or whales? Or maybe you'd like to know more about the ocean's ecosystem, like the importance of phytoplankton or the role of ocean currents? Let me know and I'll do my best to explain it in a way that's easy to understand!<|eot_id|>
