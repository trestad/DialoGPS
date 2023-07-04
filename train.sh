CUDA_VISIBLE_DEVICES=6,7, fairseq-train dd_dataset -a CSDA_MTM --optimizer adam --lr 1.5e-4 --label-smoothing 0.1 --dropout 0.1 --min-lr 1e-09 --lr-scheduler inverse_sqrt --weight-decay 1e-9 --criterion csda_ce --max-update 50000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' --save-dir mtm_folder/exp_mtm_sentence_mean_as_latent_K2_bsz_160_lr_1.5_bothdiv1_ckps --required-batch-size-multiple 1 --share-all-embeddings --task csda  --max-source-positions 512 --max-target-positions 512 --activation-dropout 0.1 --attention-dropout 0.1 --no-epoch-checkpoints --log-format json --log-interval 20  --skip-invalid-size-inputs-valid-test --best-checkpoint-metric ppl   --batch-size 10 --update-freq 8 --K 2 --eou 5 --z pre  --patience 15 