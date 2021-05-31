import ctypes
import heapq
import json
import math
import os
import random
import time
from itertools import chain
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import sys

import torch.multiprocessing as torch_mp
import multiprocessing

from pytorch_pretrained_bert import BertAdam
from table_bert.vertical.vertical_attention_table_bert import VerticalAttentionTableBert

from nsm import nn_util
from nsm.parser_module import get_parser_agent_by_name
from nsm.parser_module.dds_agent import PGAgent
from nsm.consistency_utils import ConsistencyModel, QuestionSimilarityModel
from nsm.retrainer import Retrainer, load_nearest_neighbors
from nsm.dds_evaluator import Evaluation
from nsm.program_cache import SharedProgramCache

import torch
from tensorboardX import SummaryWriter

from nsm.dist_util import STOP_SIGNAL
from nsm.sketch.sketch_predictor import SketchPredictor
from nsm.sketch.trainer import SketchPredictorTrainer


class Learner(torch_mp.Process):
    def __init__(self, config: Dict, devices: Union[List[torch.device], torch.device], dev_file: str,
         shared_program_cache: SharedProgramCache = None):
        super(Learner, self).__init__(daemon=True)



        self.train_queue = multiprocessing.Queue()
        self.categories = ['real', 'syn_agg', 'syn_sel', 'syn_rank']
        # self.queues = dict()
        # for cat in self.categories:
        #     self.queues[cat] = multiprocessing.Queue()

        self.checkpoint_queue = multiprocessing.Queue()
        self.psi_queue = multiprocessing.Queue()
        self.config = config
        self.devices = devices
        self.dev_file = dev_file
        self.dev_environments = None
        self.actor_message_vars = []
        self.current_model_path = None
        self.shared_program_cache = shared_program_cache
        self.current_psi = np.ones((len(self.categories)), dtype=np.float32)
        self.actor_num = 0


    def load_dev_environments(self):
        from table.dds_experiments import load_environments
        envs = load_environments([self.dev_file],
                                table_file=self.config['table_file'],
                                table_representation_method=self.config['table_representation'],
                                bert_tokenizer=self.agent.encoder.bert_model.tokenizer)
        for env in envs:
            env.use_cache = False
            env.punish_extra_work = False

        self.dev_environments = envs


    def run(self):
        # initialize cuda context
        devices = self.devices if isinstance(self.devices, list) else [self.devices]
        self.devices = [torch.device(device) for device in devices]

        if 'cuda' in self.devices[0].type:
            torch.cuda.set_device(self.devices[0])

        # seed the random number generators
        for device in self.devices:
            nn_util.init_random_seed(self.config['seed'], device)

        agent_name = self.config.get('parser', 'vanilla')
        self.agent = get_parser_agent_by_name(agent_name).build(self.config, master='learner').to(self.devices[0]).train()
        
        model_path = os.path.join(self.config['work_dir'], 'model.best.bin')
        if os.path.exists(model_path):
            print("Loading from saved model.")
            self.agent.load(model_path)
        else:
            print("No model found at %s." % model_path)
        # checkload s= input('Check if model loaded.')
        use_trainable_sketch_predictor = self.config.get('use_trainable_sketch_predictor', False)
        if use_trainable_sketch_predictor:
            assert len(self.devices) > 1
            if 'cuda' in self.devices[1].type:
                torch.cuda.set_device(self.devices[1])

            self.sketch_predictor = SketchPredictor.build(self.config).train().to(self.devices[1])
            self.sketch_predictor_trainer = SketchPredictorTrainer(
                self.sketch_predictor, self.config['max_train_step'], 0, self.config
            )

        self.psi_queue.put(self.current_psi)
        self.train()

    def train(self):
        model = self.agent
        config = self.config
        work_dir = Path(config['work_dir'])
        train_iter = 0
        save_every_niter = config['save_every_niter']
        entropy_reg_weight = config['entropy_reg_weight']
        summary_writer = SummaryWriter(os.path.join(config['work_dir'], 'tb_log/train'))
        max_train_step = config['max_train_step']
        save_program_cache_niter = config.get('save_program_cache_niter', 0)
        freeze_bert_for_niter = config.get('freeze_bert_niter', 0)
        gradient_accumulation_niter = config.get('gradient_accumulation_niter', 1)
        use_trainable_sketch_predictor = self.config.get('use_trainable_sketch_predictor', False)

        bert_params = [
            (p_name, p)
            for (p_name, p) in model.named_parameters()
            if 'bert_model' in p_name and p.requires_grad
        ]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_grouped_parameters = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        bert_optimizer = BertAdam(
            bert_grouped_parameters,
            lr=self.config['bert_learning_rate'],
            warmup=0.1,
            t_total=max_train_step)

        # non bert parameters
        other_params = [
            p
            for n, p
            in model.named_parameters()
            if 'bert_model' not in n and p.requires_grad
        ]

        other_optimizer = torch.optim.Adam(other_params, lr=0.001)

        # eval batch loader
        self.load_dev_environments()
        dev_iter = nn_util.loop_iter(self.dev_environments, batch_size=self.config['batch_size'], shuffle=True)

        cum_loss = cum_examples = 0.
        t1 = time.time()



        while train_iter < max_train_step:
            if 'cuda' in self.devices[0].type:
                torch.cuda.set_device(self.devices[0])

            train_iter += 1
            other_optimizer.zero_grad()
            bert_optimizer.zero_grad()


            train_samples, samples_info = self.train_queue.get()
            print("len of train examples taken from queue and len of categoriesß: ", len(train_samples), len(samples_info['category']))
            sys.stdout.flush()

            sample_categories = samples_info['category']
            dev_batched_envs = next(dev_iter)
            
            dev_samples = model.decode_examples(dev_batched_envs, beam_size=self.config['beam_size'])
            print(dev_samples)
            dev_samples = dev_samples[0]
            
            print('len dev batched envs: ', len(dev_batched_envs), len(dev_samples))
            print('train examples : ', len(train_samples))
            print('train samples variable: ', train_samples)
            print('dev samples variable', dev_samples)

            # exit()
            # how to get a sample from dev set(?)
            # dev_samples, dev_samples_info = self.dev
            # inference on dev set(?)
            # decode_results = self.agent.decode_examples(self.dev_environments, beam_size=self.config['beam_size'], batch_size=32)
            # eval_results = Evaluation.evaluate_decode_results(self.dev_environments, decode_results)

            # compute gradient wrt eval results (?)


            try:
                queue_size = self.train_queue.qsize()
                # queue_sizes = []
                # for cat in self.categories:
                #     queue_sizes.append(self.queues[cat].qsize())
                print(f'[Learner] train_iter={train_iter} train queue size={queue_size}', file=sys.stderr)
                summary_writer.add_scalar('train_queue_sizes', queue_size, train_iter)
            except NotImplementedError:
                pass

            train_trajectories = [sample.trajectory for sample in train_samples]
            
            # dev
            dev_trajectories  = [sample.trajectory for sample in dev_samples]


            # repeat for getting dev grad
            dev_loss, dev_log_prob = self.forward_single(dev_samples, train_iter, summary_writer, batch_type='dev')
            other_optimizer.step()
            grad_dev_nested = [p.grad for p in other_params]
            grad_dev = [torch.flatten(g) for g in grad_dev_nested]
    
            grad_dev = torch.cat(grad_dev)
            # grad_dev = torch.

            print('dev gradient: ', len(grad_dev), grad_dev[0])
            print('log pr dev: ', dev_log_prob)

            other_optimizer.zero_grad()
            bert_optimizer.zero_grad()
            
            # to save memory, for vertical tableBERT, we partition the training trajectories into small chunks
            # if isinstance(self.agent.encoder.bert_model, VerticalAttentionTableBert) and 'large' in self.agent.encoder.bert_model.config.base_model_name:
            #     print('233 if')
            #     chunk_size = 5
            #     # dev_chunk_size = 5
            # else:
            #     print('237 else')
            #     chunk_size = len(train_samples)
            #     dev_chunk_size = len(dev_samples)
                
            
            chunk_size = 1000000000
            chunk_num = int(math.ceil(len(train_samples) / chunk_size))
            cum_loss = 0.
            log_pr_catwise_train = torch.zeros((len(self.categories), 1))
            
            if chunk_num > 1:
                for chunk_id in range(0, chunk_num):
                    train_samples_chunk = train_samples[chunk_size * chunk_id: chunk_size * chunk_id + chunk_size]
                    sample_categories_chunk = sample_categories[chunk_size*chunk_id:chunk_size * chunk_id + chunk_size]
                    for idx, cat in enumerate(self.categories):
                        cat_indices = [j for j in range(len(train_samples_chunk)) if sample_categories_chunk[j] == cat]
                        train_cat_chunk = [train_samples_chunk[j] for j in cat_indices]
                        loss_val, log_pr_chunk = self.forward_single(train_cat_chunk, train_iter, summary_writer, batch_type='train')
                        cum_loss += loss_val
                        grad_cat = [p.grad for p in other_params]
                        
                        print('train gradient cat: ', cat, len(grad_cat))
                        print('log pr: ', log_pr_chunk)
                        reward = torch.dot(torch.tensor(grad_dev), torch.tensor(grad_cat))
                        self.current_psi[idx] = self.current_psi[idx] + self.config['dds_lr']*reward*log_pr_chunk

                
                grad_multiply_factor = 1 / len(train_samples)
                for p in self.agent.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(grad_multiply_factor)
            else:
                
                for idx, cat in enumerate(self.categories):
                        cat_indices = [j for j in range(len(train_samples)) if sample_categories[j] == cat]
                        train_cat = [train_samples[j] for j in cat_indices]
                        if not train_cat: # empty list, no samples from this category
                            print('no samples in current batch for: ', cat)
                            sys.stdout.flush()
                            continue
                        loss_val, log_pr = self.forward_single(train_cat, train_iter, summary_writer, batch_type='train')
                        cum_loss = loss_val * len(train_samples)
                        grad_cat = [p.grad for p in other_params] # ignore bert_params
                        grad_cat = [torch.flatten(g) for g in grad_cat]
                        grad_cat = torch.cat(grad_cat)
                        other_optimizer.step()
                        other_optimizer.zero_grad()
                        # for every cat, fresh gradients
                        print('train gradient cat: ', cat, len(grad_cat), grad_cat[0])
                        print('log pr: ', log_pr)
                        print(type(grad_cat), grad_cat, grad_cat.shape)
                        print(type(grad_dev), grad_dev, grad_dev.shape)
                        sys.stdout.flush()
                        
                        # t1 = torch.FloatTensor(grad_dev)
                        # t2 = torch.FloatTensor(grad_cat)
                        # print(t1.shape)
                        # sys.stdout.flush()
                        # print(t2.shape)
                        # sys.stdout.flush()
                        reward = torch.dot(grad_dev, grad_cat) / (torch.norm(grad_cat)*torch.norm(grad_dev))
                        print('reward: ', reward)
                        sys.stderr.flush()
                        sys.stdout.flush()
                        self.current_psi[idx] = self.current_psi[idx] + self.config['dds_lr']*reward*log_pr

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(other_params, 5.)

            if train_iter % gradient_accumulation_niter == 0:
                other_optimizer.step()

                if train_iter > freeze_bert_for_niter:
                    bert_optimizer.step()
                elif train_iter == freeze_bert_for_niter:
                    print(f'[Learner] train_iter={train_iter} reset Adam optimizer and start fine-tuning BERT')
                    other_optimizer = torch.optim.Adam(other_params, lr=0.001)

            
            self.psi_queue.put(self.current_psi)

            if 'clip_frac' in samples_info:
                summary_writer.add_scalar('sample_clip_frac', samples_info['clip_frac'], train_iter)

            # update sketch predictor
            if use_trainable_sketch_predictor:
                if 'cuda' in self.devices[1].type:
                    torch.cuda.set_device(self.devices[1])

                self.sketch_predictor_trainer.step(train_trajectories, train_iter=train_iter)

            cum_examples += len(train_samples)

            self.try_update_model_to_actors(train_iter)

            if train_iter % save_every_niter == 0:
                print(f'[Learner] train_iter={train_iter} avg. loss={cum_loss / cum_examples}, '
                      f'{cum_examples} examples ({cum_examples / (time.time() - t1)} examples/s)', file=sys.stderr)
                cum_loss = cum_examples = 0.
                t1 = time.time()

                # log stats of the program cache
                program_cache_stat = self.shared_program_cache.stat()
                summary_writer.add_scalar(
                    'avg_num_programs_in_cache',
                    program_cache_stat['num_entries'] / program_cache_stat['num_envs'],
                    train_iter
                )
                summary_writer.add_scalar(
                    'num_programs_in_cache',
                    program_cache_stat['num_entries'],
                    train_iter
                )

            if save_program_cache_niter > 0 and train_iter % save_program_cache_niter == 0:
                program_cache_file = work_dir / 'log' / f'program_cache.iter{train_iter}.json'
                program_cache = self.shared_program_cache.all_programs()
                json.dump(
                    program_cache,
                    program_cache_file.open('w'),
                    indent=2
                )

    def try_update_model_to_actors(self, train_iter):
        save_every_niter = self.config.get('save_every_niter')
        
        if train_iter % save_every_niter == 0:
            self.update_model_to_actors(train_iter)
        else:
            self.push_new_model(self.current_model_path)

    def forward_single(self, samples, train_iter, summary_writer, reduction='sum',
                        batch_type='train'):

        trajectories = [sample.trajectory for sample in samples]
        # (batch_size)
        print('trajectories', trajectories)
        batch_log_prob, meta_info = self.agent(trajectories, return_info=True)
    
        if batch_type=='train':
            train_sample_weights = batch_log_prob.new_tensor([s.weight for s in samples])
            batch_log_prob = batch_log_prob * train_sample_weights

        if reduction == 'sum':
            loss = - batch_log_prob.sum()
        elif reduction == 'mean':
            loss = - batch_log_prob.mean()
        else:
            raise ValueError(f'Unknown reduction {reduction}')

        gradient_accumulation_niter = self.config.get('gradient_accumulation_niter', 1)

        if gradient_accumulation_niter > 1:
            loss /= gradient_accumulation_niter

        summary_writer.add_scalar('parser_loss', loss.item(), train_iter)

        loss.backward()
#       
        loss_val = loss.item()
        return loss_val, batch_log_prob.mean()


    def train_step(self, train_samples, dev_samples, dev_trajectories,
                            sample_categories, train_iter, summary_writer, reduction='sum'):
        
        train_trajectories = [sample.trajectory for sample in train_samples]
        # (batch_size)
        batch_log_prob, meta_info = self.agent(train_trajectories, return_info=True)
        dev_log_prob, dev_meta_info = self.agent(dev_trajectories, return_info=True)
    
        train_sample_weights = batch_log_prob.new_tensor([s.weight for s in train_samples])
        batch_log_prob = batch_log_prob * train_sample_weights

        # let's just introduce all the prob calculations inside the train step.

        if reduction == 'sum':
            loss = - batch_log_prob.sum()
            dev_loss = - dev_log_prob.sum()
        elif reduction == 'mean':
            loss = - batch_log_prob.mean()
            dev_loss = - dev_log_prob.mean()
        else:
            raise ValueError(f'Unknown reduction {reduction}')

        gradient_accumulation_niter = self.config.get('gradient_accumulation_niter', 1)

        if gradient_accumulation_niter > 1:
            loss /= gradient_accumulation_niter

        summary_writer.add_scalar('parser_loss', loss.item(), train_iter)
        # loss = -batch_log_prob.sum() / max_batch_size

        loss.backward()
        print('grad loss', loss.grad)
        g_t = loss.grad
        d_p = dev_loss.grad
        print('grad dev loss', dev_loss.grad)

        rew = torch.dot(g_t, d_p)

        cat_indices = dict()
        for cat in self.categories:
            cat_indices[cat] = [j for j in range(len(train_samples)) if sample_categories[j] == cat]

        log_pr_catwise = np.zeros((len(self.categories), 1))
        for i, cat in enumerate(self.categories):
            log_pr_catwise[i] = sum([batch_log_prob[j] for j in cat_indices[cat]])

        # update equation for psi
        self.current_psi = self.current_psi - self.config['dds_lr']*rew*log_pr_catwise
        print('new psi:', self.current_psi)
        self.psi_queue.put(self.current_psi)
        # exit()
        loss_val = loss.item()

        return loss_val



    def forward_step(self, samples, sample_categories, train_iter, summary_writer, reduction='sum', batch_type='train'):
        
        trajectories = [sample.trajectory for sample in samples]
        # (batch_size)
        batch_log_prob, meta_info = self.agent(trajectories, return_info=True)
    
        if batch_type=='train':
            train_sample_weights = batch_log_prob.new_tensor([s.weight for s in samples])
            batch_log_prob = batch_log_prob * train_sample_weights


        if reduction == 'sum':
            loss = - batch_log_prob.sum()
        elif reduction == 'mean':
            loss = - batch_log_prob.mean()
        else:
            raise ValueError(f'Unknown reduction {reduction}')

        gradient_accumulation_niter = self.config.get('gradient_accumulation_niter', 1)

        if gradient_accumulation_niter > 1:
            loss /= gradient_accumulation_niter

        summary_writer.add_scalar('parser_loss', loss.item(), train_iter)
        # loss = -batch_log_prob.sum() / max_batch_size

        loss.backward()
#         print('grad loss', loss.grad)
#         g_t = loss.grad
#         d_p = dev_loss.grad
#         print('grad dev loss', dev_loss.grad)

        if batch_type == 'train':
            cat_indices = dict()
            for cat in self.categories:
                cat_indices[cat] = [j for j in range(len(samples)) if sample_categories[j] == cat]

            log_pr_catwise = torch.zeros((len(self.categories), 1))
            
            print('batch_log_prob: ', batch_log_prob)
            print('sample categories: ', sample_categories)
            for i, cat in enumerate(self.categories):
                log_pr_catwise[i] = sum([batch_log_prob[j] for j in cat_indices[cat]])
        else:
            log_pr_catwise = batch_log_prob
#         # update equation for psi
#         self.current_psi = self.current_psi - self.config['dds_lr']*rew*log_pr_catwise
#         print('new psi:', self.current_psi)
#         self.psi_queue.put(self.current_psi)
#         # exit()
        loss_val = loss.item()
        return loss_val, log_pr_catwise

    
    def update_model_to_actors(self, train_iter):
        t1 = time.time()
        model_state = self.agent.state_dict()
        model_save_path = os.path.join(self.config['work_dir'], 'agent_state.iter%d.bin' % train_iter)
        torch.save(model_state, model_save_path)

        if hasattr(self, 'sketch_predictor_server_msg_val'):
            sketch_predictor_path = str(Path(model_save_path).with_suffix('.sketch_predictor.bin'))
            torch.save(self.sketch_predictor.state_dict(), sketch_predictor_path)
        else:
            sketch_predictor_path = None

        self.push_new_model(model_save_path, sketch_predictor_path=sketch_predictor_path)
        print(f'[Learner] pushed model [{model_save_path}] (took {time.time() - t1}s)', file=sys.stderr)
        if sketch_predictor_path:
            print(f'[Learner] pushed sketch prediction model [{sketch_predictor_path}] (took {time.time() - t1}s)', file=sys.stderr)

        if self.current_model_path:
            os.remove(self.current_model_path)
            sketch_predictor_server_msg_val = getattr(self, 'sketch_predictor_server_msg_val', None)
            if sketch_predictor_server_msg_val:
                os.remove(str(Path(self.current_model_path).with_suffix('.sketch_predictor.bin')))

        self.current_model_path = model_save_path

    def push_new_model(self, model_path, sketch_predictor_path=None):

        # put new psi into the queue so all actors get access to it.
        self.psi_queue.put(self.current_psi)

        self.checkpoint_queue.put(model_path)
        if model_path:
            self.eval_msg_val.value = model_path.encode()

            table_bert_server_msg_val = getattr(self, 'table_bert_server_msg_val', None)
            if table_bert_server_msg_val:
                table_bert_server_msg_val.value = model_path.encode()

        if sketch_predictor_path:
            sketch_predictor_server_msg_val = getattr(self, 'sketch_predictor_server_msg_val', None)
            if sketch_predictor_server_msg_val:
                sketch_predictor_server_msg_val.value = sketch_predictor_path.encode()

    def register_actor(self, actor):
        actor.checkpoint_queue = self.checkpoint_queue
        # actor.queues = self.queues
        actor.train_queue = self.train_queue
        actor.psi_queue = self.psi_queue
        actor.current_psi = self.current_psi
        self.actor_num += 1

    def register_evaluator(self, evaluator):
        msg_var = multiprocessing.Array(ctypes.c_char, 4096)
        self.eval_msg_val = msg_var
        evaluator.message_var = msg_var

    def register_table_bert_server(self, table_bert_server):
        msg_val = multiprocessing.Array(ctypes.c_char, 4096)
        self.table_bert_server_msg_val = msg_val
        table_bert_server.learner_msg_val = msg_val

    def register_sketch_predictor_server(self, sketch_predictor_server):
        msg_val = multiprocessing.Array(ctypes.c_char, 4096)
        self.sketch_predictor_server_msg_val = msg_val
        sketch_predictor_server.learner_msg_val = msg_val
