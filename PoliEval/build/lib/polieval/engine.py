from __future__ import absolute_import, division, unicode_literals

import copy
from PoliEval.polieval import utils
from PoliEval.polieval.frame import FrameEval
from PoliEval.polieval.stance import StanceEval
from PoliEval.polieval.vote_select import VoteSeqEval
from PoliEval.polieval.bias import BIASSeqEval
from PoliEval.polieval.grade import GradeSeqEval

class SE(object):
    def __init__(self, params, encoder):
        # parameters
        params = utils.dotdict(params)
        params.seed = 42 if 'seed' not in params else params.seed

        params.batch_size = 32 if 'batch_size' not in params else params.batch_size
        params.epochs = 10 if 'epochs' not in params else params.epochs
        params.lr = 2e-5 if 'lr' not in params else params.lr
        params.warmup_ratio = 0 if 'warmup_ratio' not in params else params.warmup_ratio
        params.weight_decay = 1e-4 if 'weight_decay' not in params else params.weight_decay
        params.max_len = 256  if 'max_len' not in params else params.max_len
        params.model_name_or_path = 'bert-base-uncased' if 'model_name_or_path' not in params else params.model_name_or_path

        self.params = params
        self.encoder = encoder
        #
        self.list_tasks = ['PUB_STANCE_poldeb','LEG_BIAS_cong',]
        self.nclasses = {
            'PUB_STANCE_poldeb':2,
            'LEG_BIAS_cong':2,
        }

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results       

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)
        self.params.current_task = name
        self.params.nclasses = self.nclasses[name]

        # PoliEval tasks
        if name == 'MEDIA_FRAME_gvfc':
            self.params.thred = 0.5
            self.evaluation = FrameEval(tpath + '/MEDIA_FRAME/gvfc/', self.params, copy.deepcopy(self.encoder))
        elif name == 'PUB_STANCE_poldeb':
            self.evaluation = StanceEval(tpath + '/PUB_STANCE/poldeb/',  self.params, copy.deepcopy(self.encoder))
        elif name == 'LEG_BIAS_cong':
            self.evaluation = BIASSeqEval(tpath + '/PUB_BIAS/cong_records/', self.params, copy.deepcopy(self.encoder))            
        elif name == 'GRADE_LCV':
            self.evaluation = GradeSeqEval(tpath + '/GRADE/lcv/', self.params, copy.deepcopy(self.encoder))                               
        elif name == 'VOTE_out':
            self.evaluation = VoteSeqEval(tpath + '/VOTE/out-of-session/2015/', self.params, copy.deepcopy(self.encoder))                  
        
        print('Current eval task: ', name)
        self.results = self.evaluation.run()

        return self.results