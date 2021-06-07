import json
import csv
import os
import numpy as np
import distutils
from distutils import dir_util

#TODO: replace the usage of these variables with argparse counterparts
ops = 'all' # 'select' or 'all'
num_where = 4  # number of where clauses

def question_quality_check(question):
    wh_list = ['what', 'when', 'how', 'where', 'why', 'name', 'get', 'which', 'tell', 'who']
    check = False
    for wh in wh_list:
        if wh in question.lower():
            check = True
    return check

def filter_check_sql(sql):
    if ops == 'select':
        if sql['col'][0] != 'select':
            return False
    if len(sql['conds']) > num_where:
        return False
    return True


def get_wtq_qg_data(group_id='', qg_file = '',filter=['select']):
    # folder_path = '/dccstor/cmv/saneem/nlqTable/irl_git/QG-tableQA/data/generated_question/'
    folder_path = '/mnt/infonas/data/anshkhurana/table_qa/pytorch_neural_symbolic_machines/data/wikitable-ts_data/synthetic_data/'
    print(group_id, folder_path, qg_file)
    # if group_id != '':
    #     fn = folder_path + 'wtq_gen_quest_' + str(group_id) + '_.json'
    # else:
    #     fn = folder_path + qg_file
    fn = os.path.join(folder_path, qg_file)
    print('get_wtq_gq_data: ', fn)
    with open(fn) as fp:
        qg_data = json.load(fp)
    synth_data = []
    for i,data in enumerate(qg_data):
        if filter_check_sql(data['sql']):
            data['question'][0] = data['question'][0].replace('\t',' ').replace('\n',' ')
            data['sql']['table_id'] = data['sql']['table_id'].replace('\t',' ').replace('\n',' ')
            data['sql']['answer'] = data['sql']['answer'].replace('\t',' ').replace('\n',' ')
            row = ['sy-'+str(i), data['question'][0], data['sql']['table_id'], data['sql']['answer']]
            synth_data.append(row)
    return synth_data


def get_wtq_qg_tagged_data(group_id='', qg_file=''):
    folder_path = '/mnt/infonas/data/anshkhurana/table_qa/pytorch_neural_symbolic_machines/data/wikitable-ts_data/synthetic_with_types'
    qg_file_x = qg_file.replace('_ppl_score','')
    fn = os.path.join(folder_path, qg_file_x.replace('.json','.tsv'))

    with open(fn) as fp:
        reader = csv.reader(fp, delimiter='\t')
        header = reader.__next__()
        tagged_data = []
        for row in reader:
            q = row[1]
            if question_quality_check(q):
                tagged_data.append(row)
    return tagged_data

def align_tagged_and_genq(group_id='', qg_file=''):
    # if group_id != '':
    #     synth_data = get_wtq_qg_data(group_id=group_id)
    #     tagged_data = get_wtq_qg_tagged_data(group_id=group_id)    
    # else:
    synth_data = get_wtq_qg_data(qg_file=qg_file)
    tagged_data = get_wtq_qg_tagged_data(qg_file=qg_file)    

    match_tagged = []
    no_match_synth = []
    for i,d in enumerate(synth_data):
        found = False
        for j,t in enumerate(tagged_data):
            if d[1].replace('"',"'") == t[1] and d[2] == '/'.join(t[2].split('/')[-3:]):
                synth_data[i][0] = t[0]
                found = True
                match_tagged.append(j)
                break
        if not found:
            # print('Cannot match ', i)
            no_match_synth.append(i)
    print('count match ', len(no_match_synth), 'instances')
    synth_data = [s for i,s in enumerate(synth_data) if i not in no_match_synth]
    tagged_data = [tagged_data[i] for i in match_tagged]
    
    return synth_data, tagged_data

def append_synth_data(group_id, train_synth_frac = 0.2, dev_synth_frac=0.0):
    # creating a new folder to store appended data
    base_dir = '/dccstor/cmv/saneem/nlqTable/irl_git/neural-symbolic-machines/data/wikitable/raw_input_folder/raw_input-LO_'+group_id
    new_dir = '/dccstor/cmv/saneem/nlqTable/irl_git/neural-symbolic-machines/data/wikitable/'+\
        'raw_input_folder/raw_input-LO_'+group_id + '_synth_tf-' + str(train_synth_frac) + '_df-' + str(dev_synth_frac)
    data_relative_path = '/WikiTableQuestions/data/' 

    os.system('mkdir ' + new_dir)
    os.system('cp -R ' + base_dir + '/. ' + new_dir)

    # Appending synth to tagged data
    with open(base_dir + '/WikiTableQuestions/tagged/data/training.tagged') as fp:
        reader = csv.reader(fp, delimiter='\t')
        tag_header = reader.__next__()
        tagged_data = [row for row in reader]
    num_tagged = len(tagged_data)
    tagged_synth_count = round(num_tagged * train_synth_frac / (1 - train_synth_frac))

    synth_data, synth_tagged_data = align_tagged_and_genq(group_id)
    synth_data = synth_data[:tagged_synth_count]
    synth_tagged_data = synth_tagged_data[:tagged_synth_count]

    # removing context extension in synth data
    for i,s in enumerate(synth_tagged_data):
        context_list = s[2].split('/')
        context_list[-1] = context_list[-1].replace('tsv', 'csv')
        context = '/'.join(context_list[-3:])
        synth_tagged_data[i][2] = context

    all_tagged = [tag_header] + tagged_data + synth_tagged_data
    with open(new_dir + '/WikiTableQuestions/tagged/data/training.tagged', 'w') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerows(all_tagged)
    print('Tagged data appended')

    for si in ['1','2','3','4','5']:
        with open(base_dir + data_relative_path + 'random-split-'+si+'-train.tsv') as fp:
            reader = csv.reader(fp, delimiter='\t')
            header = reader.__next__()
            train_real_data = [row for row in reader]    
        num_train_real = len(train_real_data)

        with open(base_dir + data_relative_path + 'random-split-'+si+'-dev.tsv') as fp:
            reader = csv.reader(fp, delimiter='\t')
            header = reader.__next__()
            dev_real_data = [row for row in reader]    
        num_dev_real = len(dev_real_data)

        train_synth_count = round(num_train_real * train_synth_frac / (1 - train_synth_frac))
        dev_synth_count = round(num_dev_real * dev_synth_frac / (1 - dev_synth_frac))
        
        num_synth = len(synth_data)
        
        if train_synth_count < num_synth:
            sample_idx = np.random.choice(range(num_synth), train_synth_count, replace=False)
        else:
            sample_idx = list(range(num_synth)) * int(np.floor(train_synth_count/num_synth))
            sample_idx += list(np.random.choice(range(num_synth), train_synth_count%num_synth, replace=False))
        sampled_synth = [synth_data[i] for i in sample_idx]
        
        # ACL submission quick hacks
        # all_train = [header] + train_real_data + sampled_synth
        all_train = [header] + train_real_data + synth_data

        if dev_synth_count < num_synth:
            sample_idx = np.random.choice(range(num_synth), dev_synth_count, replace=False)
        else:
            sample_idx = list(range(num_synth))*(dev_synth_count/num_synth)
            sample_idx += np.random.choice(range(num_synth), dev_synth_count % num_synth, replace=False)
        sampled_synth = [synth_data[i] for i in sample_idx]
        all_dev = [header] + dev_real_data + sampled_synth

        with open(new_dir + data_relative_path + 'random-split-'+si+'-train.tsv', 'w') as fp:
            writer = csv.writer(fp, delimiter='\t')
            writer.writerows(all_train)
        with open(new_dir + data_relative_path + 'random-split-'+si+'-dev.tsv', 'w') as fp:
            writer = csv.writer(fp, delimiter='\t')
            writer.writerows(all_dev)
    print('Data created' + new_dir)


def align_generated_questions(genq1, genq2):
    genq2_reordered = []
    genq2_dict = {}
    for g in genq2:
        genq2_dict[str(g['sql'])] = g
    for g in genq1:
        genq2_reordered.append(genq2_dict[str(g['sql'])])

    return genq2_reordered


def tag_questions_using_sempre():
    qg_folder = '../QG-tableQA/data/lisp_format/'
    qg_files = os.listdir(qg_folder)

    for i,qg in enumerate(qg_files):
        tag_command =  """./run @mode=tables @class=tag-data \
            -dataset.inpaths anygroupname:../QG-tableQA/data/lisp_format/{} \
            -baseCSVDir ../QG-tableQA/data/WikiTableQuestions/ @useTaggedFile=0""".format(qg)
        mv_command = 'mv state/execs/0.exec/tagged-anygroupname.tsv ../QG-tableQA/data/tagged/'+qg.replace('.examples','.tsv')
        rm_command1 = "rm -rf state/execs"
        rm_command2 = "rm -rf state/lastExec"
        
        
        os.system(tag_command)
        os.system(mv_command)
        os.system(rm_command1)
        os.system(rm_command2)
        print(i,qg)

def synth_filter_from_processed_input(group_id, synth_train_frac=0.3, synth_dev_frac=0.2, lookup=False):
    lookup_prefix = ''
    if not lookup:
        base_dir = '/dccstor/cmv/saneem/nlqTable/irl_git/neural-symbolic-machines/data/wikitable/' +\
            'processed_input/processed_input-LO_'+group_id+'_synth_tf-0.9_df-0.0/'

    else:
        base_dir = '/dccstor/cmv/saneem/nlqTable/irl_git/neural-symbolic-machines/data/wikitable/' +\
            'processed_input/processed_input-LO_'+group_id+'_synth_lookup_tf-0.9_df-0.0/'
        lookup_prefix = '_lookup'
    
    new_dir = '/dccstor/cmv/saneem/nlqTable/irl_git/neural-symbolic-machines/data/wikitable/' +\
            'processed_input/processed_input-LO_'+group_id+'_filter_synth_cleaned' + lookup_prefix +\
            '_tf-' + str(synth_train_frac) + '_df-' + str(synth_dev_frac) + '/'

    os.system('mkdir ' + new_dir)
    os.system('cp -R ' + base_dir + '/. ' + new_dir)

    with open(base_dir + 'data_split_1/train_split.jsonl') as fp:
        train_split = [json.loads(t) for t in fp.readlines()]
    synthetic_ids = [t['id'] for t in train_split if 'syn' in t['id']]

    with open(base_dir + 'data_split_1/dev_split.jsonl') as fp:
        dev_split = [json.loads(t) for t in fp.readlines()]
    num_dev = len(dev_split)
    
    num_synth = len(synthetic_ids)
    num_real = len(train_split) - len(synthetic_ids)
    num_train_synth = round(num_real * synth_train_frac / (1 - synth_train_frac))
    num_dev_synth = round(num_dev * synth_dev_frac / (1 - synth_dev_frac))
    
    print ('#### ',num_train_synth + num_dev_synth - num_synth, 'overlap', group_id)
    shuffled_syn_ids = list(np.random.permutation(synthetic_ids))
    train_syn_ids = shuffled_syn_ids[:num_train_synth]
    dev_syn_ids = shuffled_syn_ids[-num_dev_synth:]

    for filename in os.listdir(new_dir + 'data_split_1/'):
        file_path = new_dir + 'data_split_1/' + filename
        with open(file_path) as fp:
            data = [json.loads(d) for d in fp.readlines()]

        if 'dev' in filename:
            for t in train_split:
                if t['id'] in dev_syn_ids:
                    data.append(t)
        elif 'train' in  filename:
            data_subset = []
            for d in data:
                if 'syn' not in d['id'] or d['id'] in train_syn_ids:
                    data_subset.append(d)
            data = data_subset
        with open(file_path, 'w') as fp:
            data_list = [json.dumps(d) for d in data]
            fp.write('\n'.join(data_list))
        
        print('Filtering', filename, 'Done')

def check_and_make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



def create_synth_data(qg_file, base_dir, synth_tagged, dev_fraction, new_prefix):
    pass


# Combine synthetic and real dataset:
def combine_synth_and_real(qg_file, base_dir, complete_real_tagged, 
                            train_synth_frac = 0.2, dev_synth_frac=0.0, new_prefix=''):

    group_id = 'g_' + qg_file.split('_g_')[1][0]
    qg_name = qg_file.split(group_id + '_')[1].replace('.json','')

    # creating a new folder to store appended data
    # base_dir = '/dccstor/cmv/saneem/nlqTable/irl_git/neural-symbolic-machines/data/wikitable/raw_input_folder/raw_input-LO_'+group_id
    base_dir = base_dir + group_id 
    # new_dir = '/dccstor/cmv/saneem/nlqTable/irl_git/neural-symbolic-machines/data/wikitable/raw_input_folder/raw_input-LO_'+group_id +\
            # '_synth_tf-' + str(train_synth_frac) + '_df-' + str(dev_synth_frac) + '_ops-' + ops + '_nw-' + str(num_where) +'_qg-' + qg_name
    new_dir = base_dir + ('_%s' % new_prefix) + '_synth_tf-' + str(train_synth_frac) + '_df-' + \
                str(dev_synth_frac) + '_ops-' + ops + '_nw-' + str(num_where) + \
                    '_qg-' + qg_name 

    data_relative_path = '/WikiTableQuestions/data/' 

    os.system('mkdir ' + new_dir)
    os.system('cp -R ' + base_dir + '/. ' + new_dir)
    
    # TODO: replace with more pythonic vesion.     
    # check_and_make_dir(new_dir)
    # distutils.dir_util.copy_tree(base_dir, new_dir)

    # Appending synth to tagged data
    with open(complete_real_tagged) as fp:
        reader = csv.reader(fp, delimiter='\t')
        tag_header = reader.__next__()
        tagged_data = [row for row in reader]
    
    num_tagged = len(tagged_data)
    synth_data, synth_tagged_data = align_tagged_and_genq(qg_file=qg_file)

    # removing context extension in synth data
    for i,s in enumerate(synth_tagged_data):
        context_list = s[2].split('/')
        context_list[-1] = context_list[-1].replace('tsv', 'csv')
        context = '/'.join(context_list[-3:])
        synth_tagged_data[i][2] = context

    train_synth_count = 0
    dev_synth_count = 0
    for si in ['1','2','3','4','5']:
        with open(base_dir + data_relative_path + 'random-split-'+si+'-train.tsv') as fp:
            reader = csv.reader(fp, delimiter='\t')
            header = reader.__next__()
            train_real_data = [row for row in reader]    
        num_train_real = len(train_real_data)

        with open(base_dir + data_relative_path + 'random-split-'+si+'-dev.tsv') as fp:
            reader = csv.reader(fp, delimiter='\t')
            header = reader.__next__()
            dev_real_data = [row for row in reader]    
        num_dev_real = len(dev_real_data)

        if train_synth_count == 0:
            train_synth_count = round(num_train_real * train_synth_frac / (1 - train_synth_frac))
            dev_synth_count = round(num_dev_real * dev_synth_frac / (1 - dev_synth_frac))
        
        num_synth = len(synth_data)
        
        # if train_synth_count < num_synth:
        #     sample_idx = np.random.choice(range(num_synth), train_synth_count, replace=False)
        # else:
        #     sample_idx = list(range(num_synth)) * int(np.floor(train_synth_count/num_synth))
        #     sample_idx += list(np.random.choice(range(num_synth), train_synth_count%num_synth, replace=False))

        # if dev_synth_count < num_synth:
        #     sample_idx = np.random.choice(range(num_synth), dev_synth_count, replace=False)
        # else:
        #     sample_idx = list(range(num_synth))*(dev_synth_count/num_synth)
        #     sample_idx += np.random.choice(range(num_synth), dev_synth_count % num_synth, replace=False)

        train_subset_synth = synth_data[:train_synth_count]
        dev_subset_synth = synth_data[train_synth_count:train_synth_count+dev_synth_count]

        if train_synth_count + dev_synth_count > len(synth_data):
            dev_subset_synth += synth_data[:train_synth_count + dev_synth_count - len(synth_data)]

        all_train = [header] + train_real_data + train_subset_synth
        all_dev = [header] + dev_real_data + dev_subset_synth
        all_tagged = [tag_header] + tagged_data + synth_tagged_data[:train_synth_count+dev_synth_count]

        
        with open(new_dir + data_relative_path + 'random-split-'+si+'-train.tsv', 'w') as fp:
            writer = csv.writer(fp, delimiter='\t')
            writer.writerows(all_train)
        with open(new_dir + data_relative_path + 'random-split-'+si+'-dev.tsv', 'w') as fp:
            writer = csv.writer(fp, delimiter='\t')
            writer.writerows(all_dev)


    with open(new_dir + '/WikiTableQuestions/tagged/data/training.tagged', 'w') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerows(all_tagged)
    
    print('Tagged data appended')
    print('Data created' + new_dir)

