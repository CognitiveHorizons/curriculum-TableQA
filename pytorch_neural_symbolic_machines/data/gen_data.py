import os
import argparse
from create_synth_plus_real_dataset import combine_synth_and_real, create_synth_data, combine_all_synth_data

parser = argparse.ArgumentParser()

parser.add_argument('--ops', type=str, default='all')
parser.add_argument('--num_where', type=int, default=4)
parser.add_argument('--dev_synth_frac', type=float, default=0.0)
parser.add_argument('--train_synth_frac', type=float, default=0.2)
parser.add_argument('--group_id', type=int, required=True)
# parser.add_argument('--gq_file', type=str, default='wtq_gen_quest_g_0_col-header__beam-10_ppl_score.json')
parser.add_argument('--base_dir', type=str, default='/mnt/infonas/data/anshkhurana/table_qa/pytorch_neural_symbolic_machines/data/wikitable-ts_data/compressed_raw_input/raw_input_folder/raw_input-LO_')
parser.add_argument('--comp_tagged', type=str, default='/mnt/infonas/data/anshkhurana/table_qa/pytorch_neural_symbolic_machines/data/downloaded/wikitable/raw_input/WikiTableQuestions/tagged/data/training.tagged')
parser.add_argument('--new_prefix', type=str, default='')
parser.add_argument('--include_real', action='store_true')


if __name__=='__main__':
    args = parser.parse_args()

    # gq_file = os.path.join(args.gq_base, 'wtq_gen_quest_g_%d_col-header__beam-10.tsv' % args.group_id)
    # gq_file = os.path.join(
    qg_file = 'wtq_gen_quest_g_%d_col-header__per_table_100.json' % args.group_id
 
    if args.include_real:
        print('creating combined data')
        combine_all_synth_data(qg_file=qg_file, base_dir=args.base_dir, complete_real_tagged=args.comp_tagged,
                            dev_synth_frac=args.dev_synth_frac,
                            new_prefix=args.new_prefix)
    else:
        create_synth_data(qg_file=qg_file, base_dir=args.base_dir, synth_tagged=args.synth_tagged,
                            dev_fraction=0.2,
                            new_prefix=args.new_prefix)