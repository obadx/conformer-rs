from conformer_python.muaalem_offline import (
    Wav2Vec2BertForMultilevelCTC,
    Wav2Vec2BertForMultilevelCTCConfig,
    vocab,
)

if __name__ == "__main__":
    config = Wav2Vec2BertForMultilevelCTCConfig.from_pretrained(
        "obadx/muaalem-model-v3_2",
        pad_token_id=vocab.PAD_TOKEN_IDX,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=True,
        num_hidden_layers=16,
        hidden_size=144,
        output_hidden_size=144,
        intermediate_size=256,
        num_attention_heads=4,
    )
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
        "obadx/muaalem-model-v3_2",
        config=config,
        ignore_mismatched_sizes=True,
    )
    print(model)
