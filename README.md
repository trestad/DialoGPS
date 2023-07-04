## DialoGPS: Dialogue Path Sampling in Continuous Semantic Space for Data Augmentation in Multi-Turn Conversations （ACL 2023）

### Install
The code is implemented based on [fairseq](https://github.com/facebookresearch/fairseq) and we refer to some contrastive learning code from [Wang et al](https://github.com/rosewang2008/language_modeling_via_stochastic_processes). We appreciate these open-source codes.

    git clone https://github.com/trestad/DialoGPS.git
    cd dialogps
    pip install -e .

### Data Preprocess
The preprocess is kind of complicated and thus we open our processed DailyDialog dataset in 'dd_dataset' for better understanding.
1.  Prepare a multi-turn dialogue dataset.
2.  For each dialogue, e.g., (u1, u2, u3, u4) where ui denotes the i-th utterance, process it to three files ends with .pre, .cxt, and .res. Here is a case showing the relationships among three files:
  > train.pre: u1 \<eou> u2 \<eou> u3 \<eou> u4 \<eou>
> 
  > train.cxt: u1 \<eou> u2 \<eou> u3 \<eou>
> 
  > train.res: u4

3. Perform the same operations on the test set and validation set.
4.  Use  `fairseq-preprocess`  to process the **training data** and obtain a vocabulary:
	```
	fairseq-preprocess  --only-source -s pre --trainpref dd_dataset/train.pre --destdir dd_dataset --workers 60
	```
	This command outputs a vocabulary in the 'dict.pre.txt' in dd_dataset.
5.  Based on the obtained vocabulary, use  `fairseq`  to convert the text data into binary data for convenient training and inference usage:
	```
	fairseq-preprocess  -s cxt -t res --trainpref dd_dataset/train --validpref dd_dataset/valid --testpref dd_dataset/test --destdir dd_dataset --workers 60 --srcdict dd_dataset/dict.pre.txt  --tgtdict dd_dataset/dict.pre.txt
	fairseq-preprocess  -s pre -t res --trainpref dd_dataset/train --validpref dd_dataset/valid --testpref dd_dataset/test --destdir dd_dataset --workers 60 --srcdict dd_dataset/dict.pre.txt  --tgtdict dd_dataset/dict.pre.txt
	```

### Training

    fairseq-train <YOUR_DATA_DIR> -a CSDA_MTM --criterion csda_ce --optimizer adam  --lr 1.0e-4 --label-smoothing 0.1 --dropout 0.1 --min-lr 1e-09 --lr-scheduler inverse_sqrt --weight-decay 1e-9  --max-update 50000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' --save-dir <YOUR_SAVE_DIR> --required-batch-size-multiple 1 --share-all-embeddings --task csda --max-source-positions 512 --max-target-positions 512 --activation-dropout 0.1 --attention-dropout 0.1 --no-epoch-checkpoints --log-format json --log-interval 1  --skip-invalid-size-inputs-valid-test --best-checkpoint-metric ppl   --batch-size <BATCH_PER_GPU> --update-freq <GRAD_ACCUMULATE>  --patience 10 --eou <EOU_IDX> --z pre --K <K> --scale <ROUGHLY 1/DELTA IN THE PAPER>

Above is an example of a training command, with some important details that need to be specified:

  1. <YOUR_DATA_DIR>: the path of preprocessed data.
  2. <YOUR_SAVE_DIR>: the path to save checkpoints.
  3. <BATCH_PER_GPU>: when you have multiple gpus, this specifies the training sample number per gpu, therefore the actual batch size is <BATCH_PER_GPU> * GPUs you used.
  4. <GRAD_ACCUMULATE>: gradient accumulate steps
  5. <EOU_IDX>: the index of the special token [eou] in your vocabulary. For example, in my dd_dataset, this value is 5. (How to check its index? As you can see in the 'dd_dataset/dict.txt', the special token '\<eou\>' is the second token. However, fairseq has 3 reserved token <\pad>, <\s>, and <\s> which does not written in the vocabulary explicitly. Therefore the actual index of '\<eou\>' is 5.)
  6. For details of other command options, please refer to the official documentation.

### Inference
This is a command to generate response with top-5 sampling:

    fairseq-generate <YOUR_DATA_DIR> --path <YOUR_SAVE_DIR>/checkpoint_best.pt --sampling-topk 5 --sampling --beam 1  > output.txt

### Evaluation
We open a DailyDialog multi-reference evaluation script in the code.
In order to facilitate comparison with us, here is a [checkpoint](https://tobedone.com) trained on DailyDialog (K=16). You can use it to try above process. 
In the repo, 'K16.output' is the output of this checkpoint, let's evaluate it:

    python multi_eval.py K16.output

Here is the evaluation output:

    dist1: 4.46
    dist2: 29.63
    Bleu_1; 38.66703642077899
    Bleu_2; 15.155684035793533
    Bleu_3; 6.98102091804927
    Bleu_4; 3.6405693431264945
To evaluate BLEURT score, please refer to the [official document](https://github.com/google-research/bleurt)
