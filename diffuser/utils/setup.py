import os
import importlib
import random
import numpy as np
import torch
from tap import Tap
import pdb

from .serialization import mkdir
from .git_utils import (
    get_git_rev,
    save_git_diff,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def watch(args_to_watch):
    def _fn(args):
        exp_name = []
        for key, label in args_to_watch:
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            if type(val) == dict:
                val = '_'.join(f'{k}-{v}' for k, v in val.items())
            exp_name.append(f'{label}{val}')
        exp_name = '_'.join(exp_name)
        exp_name = exp_name.replace('/_', '/')
        exp_name = exp_name.replace('(', '').replace(')', '')
        exp_name = exp_name.replace(', ', '-')
        return exp_name
    return _fn

def lazy_fstring(template, args):
    ## https://stackoverflow.com/a/53671539
    return eval(f"f'{template}'")

class Parser(Tap):

    # save(self): 这个方法是用于保存参数的。
    def save(self):
        # 首先创建一个文件的完整路径 fullpath
        fullpath = os.path.join(self.savepath, 'args.json')
        # 然后打印一条消息，该消息表示参数已经被保存到 fullpath 这个位置。
        print(f'[ utils/setup ] Saved args to {fullpath}')
        # 调用超类的 save 方法，保存数据到 fullpath，忽略那些不能被 pickle 序列化的对象。
        super().save(fullpath, skip_unpicklable=True)

    # 析参数的方法
    def parse_args(self, experiment=None):
        # 首先调用其超类的 parse_args 方法（只获取已知参数）
        args = super().parse_args(known_only=True)
        # 如果解析后的参数对象 args 没有 'config' 这个属性，那么就直接返回 args。
        if not hasattr(args, 'config'): return args
        # 如果有 'config' 属性，它会执行一系列操作来处理或修改参数，并最终返回处理后的参数对象。
        # 用于从配置文件中加载参数。
        args = self.read_config(args, experiment)
        # 添加额外参数的方法
        self.add_extras(args)
        # 对字符串内的表达式进行求值
        self.eval_fstrings(args)
        # 设定随机种子
        self.set_seed(args)
        #  获取提交(commit)信息的方法
        self.get_commit(args)
        # 设定加载基础(loadbase)的方法
        self.set_loadbase(args)
        # 生成实验名称的方法
        self.generate_exp_name(args)
        # 创建目录的方法
        self.mkdir(args)
        self.save_diff(args)
        return args
    
    # 所以，整体来说，该函数的目标是从指定的配置文件导入模块，并读取具体的参数配置，最后将这些配置加载到 args 对象中。
    def read_config(self, args, experiment):
        '''
            Load parameters from config file
        '''
        dataset = args.dataset.replace('-', '_')
        print(f'[ utils/setup ] Reading config: {args.config}:{dataset}')
        # 保存difference的方法，可能指保存参数或实验结果之间的不同
        module = importlib.import_module(args.config)
        # 检查 module 是否有名为 dataset 的属性，如果有，并且experiment在module.dataset.中，更新 params，否则继续
        params = getattr(module, 'base')[experiment]
        
        # 从module中获取名为 'base' 的属性，并用键为 experiment 的字典值更新 params
        if hasattr(module, dataset) and experiment in getattr(module, dataset):
            print(f'[ utils/setup ] Using overrides | config: {args.config} | dataset: {dataset}')
            overrides = getattr(module, dataset)[experiment]
            params.update(overrides)
        else:
            print(f'[ utils/setup ] Not using overrides | config: {args.config} | dataset: {dataset}')
       
        # 初始化 self._dict 为一个空字典。
        self._dict = {}
        # 然后遍历 params 的每一对键值对，将其设置为 args 的属性，并存储在 self._dict 中。
        for key, val in params.items():
            setattr(args, key, val)
            self._dict[key] = val
        # 最后返回更新后的args
        return args

    # 处理额外的命令行参数，并将它们添加到﻿args和﻿self._dict中
    def add_extras(self, args):
        '''
            Override config parameters with command-line arguments
        '''
        extras = args.extra_args
        if not len(extras):
            return

        print(f'[ utils/setup ] Found extras: {extras}')
        # 这些额外的参数在命令行中是成对提供的，故方法中有对参数数量（必须是偶数）的检查。
        assert len(extras) % 2 == 0, f'Found odd number ({len(extras)}) of extras: {extras}'
        for i in range(0, len(extras), 2):
            key = extras[i].replace('--', '')
            val = extras[i+1]
            assert hasattr(args, key), f'[ utils/setup ] {key} not found in config: {args.config}'
            old_val = getattr(args, key)
            old_type = type(old_val)
            print(f'[ utils/setup ] Overriding config | {key} : {old_val} --> {val}')
            if val == 'None':
                val = None
            elif val == 'latest':
                val = 'latest'
            elif old_type in [bool, type(None)]:
                try:
                    val = eval(val)
                except:
                    print(f'[ utils/setup ] Warning: could not parse {val} (old: {old_val}, {old_type}), using str')
            else:
                val = old_type(val)
            setattr(args, key, val)
            self._dict[key] = val

    # 处理所有在 self._dict 中以 "f:" 开头的字符串，将其转化为f-string然后解析，
    def eval_fstrings(self, args):
        for key, old in self._dict.items():
            if type(old) is str and old[:2] == 'f:':
                val = old.replace('{', '{args.').replace('f:', '')
                new = lazy_fstring(val, args)
                print(f'[ utils/setup ] Lazy fstring | {key} : {old} --> {new}')
                setattr(self, key, new)
                self._dict[key] = new

    # 控制随机数生成，为了确保每次运行实验能得到相同的结果。
    def set_seed(self, args):
        if not hasattr(args, 'seed') or args.seed is None:
            return
        print(f'[ utils/setup ] Setting seed: {args.seed}')
        set_seed(args.seed)

    # 如果 args 有一个名为 "loadbase" 的属性且其值为None，那么将它设为 "logbase"。
    def set_loadbase(self, args):
        if hasattr(args, 'loadbase') and args.loadbase is None:
            print(f'[ utils/setup ] Setting loadbase: {args.logbase}')
            args.loadbase = args.logbase

    # 如果 args有一个名为 "exp_name" 的属性，那么方法会生成一个实验名称，并将其存储在 args 和 self._dict中。
    def generate_exp_name(self, args):
        if not 'exp_name' in dir(args):
            return
        exp_name = getattr(args, 'exp_name')
        if callable(exp_name):
            exp_name_string = exp_name(args)
            print(f'[ utils/setup ] Setting exp_name to: {exp_name_string}')
            setattr(args, 'exp_name', exp_name_string)
            self._dict['exp_name'] = exp_name_string

    # 创建一个目录来储存实验结果，并保存
    def mkdir(self, args):
        if 'logbase' in dir(args) and 'dataset' in dir(args) and 'exp_name' in dir(args):
            args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name)
            self._dict['savepath'] = args.savepath
            if 'suffix' in dir(args):
                args.savepath = os.path.join(args.savepath, args.suffix)
            if mkdir(args.savepath):
                print(f'[ utils/setup ] Made savepath: {args.savepath}')
            self.save()

    # 获取当前git版本的修订编号，用于标记实验是基于代码的哪个版本进行的。
    def get_commit(self, args):
        args.commit = get_git_rev()

    # 这个方法试图保存当前工作目录和最近一次提交之间的git差异
    def save_diff(self, args):
        try:
            save_git_diff(os.path.join(args.savepath, 'diff.txt'))
        except:
            print('[ utils/setup ] WARNING: did not save git diff')
