import json
import csv
import os
import numpy as np
from shutil import copyfile
import distutils
from distutils import dir_util

tagged_data_path = 'synthetic_data_tagged'
synth_data_path = 'synthetic_data'
type_tagged_data_path = 'synthetic_with_types'
real_data_path = 'compressed_raw_input/raw_input_folder'
compete_real_tagged = '../downloaded/wikitable/raw_input/WikiTableQuestions/tagged/data/training.tagged'


def get_wtq_qg_data(fn):
    with open(fn) as fp:
        qg_data = json.load(fp)
    synth_data = []
    for i,data in enumerate(qg_data):
        data['question'][0] = data['question'][0].replace('\t',' ').replace('\n',' ')
        data['sql']['table_id'] = data['sql']['table_id'].replace('\t',' ').replace('\n',' ')
        data['sql']['answer'] = data['sql']['answer'].replace('\t',' ').replace('\n',' ')
        row = ['sy-'+str(i), data['question'][0], data['sql']['table_id'], data['sql']['answer'], data['sql']['col'][0]]
        synth_data.append(row)
    return synth_data


def question_quality_check(question):
    wh_list = ['what', 'when', 'how', 'where', 'why', 'name', 'get', 'which', 'tell', 'who']
    check = False
    for wh in wh_list:
        if wh in question.lower():
            check = True
    return check


def get_wtq_qg_tagged_data(fn):
    with open(fn) as fp:
        reader = csv.reader(fp, delimiter='\t')
        header = reader.__next__()
        tagged_data = []
        for row in reader:
            q = row[1]
            if question_quality_check(q):
                tagged_data.append(row)
    return tagged_data


def clean_up_questions(synth_data):
    question_idx = 1
    unique_questions = set()
    cleaned_data = []
    for question in synth_data:
        if question[question_idx] in unique_questions:
            continue
        else:
            unique_questions.add(question[question_idx])
            cleaned_data.append(question)
    assert len(cleaned_data) == len(unique_questions)
    print("Number of unique questions: ", len(cleaned_data))
    return cleaned_data


def align_tagged_and_genq(synth_data, tagged_data):    
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
    print('failed to match ', len(no_match_synth), 'instances')
    synth_data = [s for i,s in enumerate(synth_data) if i not in no_match_synth]
    tagged_data = [tagged_data[i] for i in match_tagged]
    
    return synth_data, tagged_data


def backup_original(real_data_path):
    for group_id in ['0', '1', '2', '3', '4']:
        tagged_data_dir = os.path.join(real_data_path, 'raw_input-LO_g_%s' % group_id, 'WikiTableQuestions', 'tagged', 'data')
        
        # make copy
        for file in os.listdir(tagged_data_dir):
            copy_file = file.split('.')[0]
            copy_file = copy_file + '_original.tagged'
            src = os.path.join(tagged_data_dir, file)
            dst = os.path.join(tagged_data_dir, copy_file)
            copyfile(src, dst)

def typify_complete_real_data(tagged_file_path):
    with open(tagged_file_path, 'r') as fp:
        reader = csv.reader(fp, delimiter='\t')
        tag_header = reader.__next__()
        tagged_data = [row for row in reader]
    
    print(len(tagged_data))
    typed_rows = []
    for df_entry in tagged_data:
        df_entry.append('real')
        df_entry.append('NA')
        typed_rows.append(df_entry)
    
    with open(tagged_file_path, 'w') as fp:
        writer = csv.writer(fp, delimiter='\t')
        tag_header.append('d_type')
        tag_header.append('op')
        # print(tag_header)
        typed_rows = [tag_header] + typed_rows
        writer.writerows(typed_rows)


# just modify the training.tagged file?
def typify_real_data(real_data_path):
    for group_id in ['0', '1', '2', '3', '4']:
        tagged_data_dir = os.path.join(real_data_path, 'raw_input-LO_g_%s' % group_id, 'WikiTableQuestions', 'tagged', 'data')
        for file in os.listdir(tagged_data_dir):
            if file.endswith('_original.tagged'):
                continue
            else:
                tagged_file = os.path.join(tagged_data_dir, file)
                with open(tagged_file, 'r') as fp:
                    reader = csv.reader(fp, delimiter='\t')
                    tag_header = reader.__next__()
                    tagged_data = [row for row in reader]
                
                typed_rows = []
                for df_entry in tagged_data:
                    df_entry.append('real')
                    df_entry.append('NA')
                    typed_rows.append(df_entry)
                
                with open(tagged_file, 'w') as fp:
                    writer = csv.writer(fp, delimiter='\t')
                    tag_header.append('d_type')
                    tag_header.append('op')
                    # print(tag_header)
                    typed_rows = [tag_header] + typed_rows
                    writer.writerows(typed_rows)

                        

def typify_syn_data(tagged_data_path, synth_data_path, type_tagged_data_path):

    for group_id in ['0', '1', '2', '3', '4']:
        tagged_file = os.path.join(tagged_data_path, 'wtq_gen_quest_g_%s_col-header__beam-10.tsv' % group_id)
        synth_file = os.path.join(synth_data_path, 'wtq_gen_quest_g_%s_col-header__beam-10_ppl_score.json' % group_id)
        typed_file = os.path.join(type_tagged_data_path, 'wtq_gen_quest_g_%s_col-header__beam-10.tsv' % group_id)


        synth_data = get_wtq_qg_data(synth_file)
        print('synth examples: ', len(synth_data))
        synth_data = clean_up_questions(synth_data)

        tagged_data = get_wtq_qg_tagged_data(tagged_file)    
        print('tagged examples: ', len(tagged_data))

        with open(tagged_file, 'r') as fp:
            reader = csv.reader(fp, delimiter='\t')
            tag_header = reader.__next__()

        synth_questions, tagged_data = align_tagged_and_genq(synth_data, tagged_data)

        
        assert(len(synth_questions) == len(tagged_data)), '%d %d' % (len(synth_questions), len(tagged_data)) 

        typed_rows = []
        for question, df_entry in zip(synth_questions, tagged_data):
            op =  question[-1]
            dtype = "syn"            
            df_entry.append(dtype)
            df_entry.append(op)
            typed_rows.append(df_entry)

        typed_rows.sort(key=lambda x: int(x[0].split('-')[-1]))
        with open(typed_file, 'w') as fp:
            writer = csv.writer(fp, delimiter='\t')
            tag_header.append('d_type')
            tag_header.append('op')
            # print(tag_header)
            typed_rows = [tag_header] + typed_rows
            writer.writerows(typed_rows)

if __name__=='__main__':
    # typify_syn_data(tagged_data_path, synth_data_path, type_tagged_data_path)
    # backup_original(real_data_path)
    # typify_real_data(real_data_path)
    typify_complete_real_data(compete_real_tagged)