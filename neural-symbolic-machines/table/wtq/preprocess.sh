DATA_DIR=$ROOT"data/wikitable/"
SUFFIX="LO_g_4_big_with_types_synth_all-_df-0.0_ops-all_nw-4_qg-col-header__per_table_100"
SOURCE="/mnt/infonas/data/anshkhurana/table_qa/pytorch_neural_symbolic_machines/data/wikitable-ts_data/compressed_raw_input/raw_input_folder/raw_input-"
DEST="/mnt/infonas/data/anshkhurana/table_qa/pytorch_neural_symbolic_machines/data/wikitable-ts_data/synthetic_preprocessed/"
RAW_DATA="$SOURCE$SUFFIX"
echo "Preprocessing data from $RAW_DATA"


python preprocess.py \
       --raw_input_dir=$RAW_DATA \
       --processed_input_dir=$DEST$SUFFIX \
       --max_n_tokens_for_num_prop=10 \
       --min_frac_for_ordered_prop=0.2 \
       --use_prop_match_count_feature \
       --expand_entities \
       --process_conjunction \
       --anonymize_datetime_and_number_entities \
       --alsologtostderr
