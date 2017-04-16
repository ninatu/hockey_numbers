from caffe.proto import caffe_pb2
import caffe
from tempfile import mkstemp
import os


class SolverProto:

    SOLVER_MODE = {
        'GPU' : caffe_pb2.SolverParameter.GPU,
        'CPU' : caffe_pb2.SolverParameter.CPU
    }

    SOLVER_TYPE = {
        'SGD': caffe_pb2.SolverParameter.SGD,
        'NESTEROV' : caffe_pb2.SolverParameter.NESTEROV,
        'ADAGRAD' : caffe_pb2.SolverParameter.ADAGRAD,
        'RMSPROP' : caffe_pb2.SolverParameter.RMSPROP,
        'ADADELTA' : caffe_pb2.SolverParameter.ADADELTA,
        'ADAM' : caffe_pb2.SolverParameter.ADAM
    }

    _defaults = {
        'base_lr': 0.01,
        'lr_policy': 'step',
        'gamma': 0.1,
        'stepsize': 100000,
        'display': 20,
        'max_iter': 450000,
        'test_iter': 4,
        'test_interval': 200,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'snapshot': 10000,
        'snapshot_prefix': "snapshot_model",
        'solver_mode': "GPU",
        'solver_type': "SGD"
    }

    def __init__(self, net_params, params):

        self._params = caffe_pb2.SolverParameter()
        self._params.net_param.MergeFrom(net_params)

        for key, value in SolverProto._defaults.items():
            params[key] = params.get(key, value)

        self._params.base_lr = params['base_lr']
        self._params.lr_policy = params['lr_policy']

        if params['lr_policy'] == 'step':
            self._params.gamma = params['gamma']
            self._params.stepsize = params['stepsize']
        elif params['lr_policy'] == 'fixed':
            pass
        else:
            raise NotImplementedError(params['lr_policy'])

	    self._params.max_iter = params['max_iter']
        self._params.momentum = params['momentum']
        self._params.weight_decay = params['weight_decay']
        self._params.display = params['display']
        self._params.test_iter = params['test_iter']
        self._params.test_interval = params['test_interval']
        self._params.snapshot = params['snapshot']
        self._params.snapshot_prefix = params['snapshot_prefix']
        self._params.solver_mode = SolverProto.SOLVER_MODE[params['solver_mode']]
        self._params.solver_type = SolverProto.SOLVER_TYPE[params['solver_type']]

        self._solver_file = mkstemp()

        print('Created solver path:', self._solver_file[1])
        with open(self._solver_file[1], 'w') as f:
            f.write(str(self._params))


    def get_path(self):
        return self._solver_file[1]


    def close(self):
        os.close(self._solver_file[0])
        os.remove(self._solver_file[1])


class Solver:

    def __init__(self, solverproto):

        self._solverproto = solverproto

    def solve(self, pretrained_path=None):

        solver = caffe.get_solver(self._solverproto.get_path())
        if pretrained_path:
            solver.net.copy_from(pretrained_path)
        solver.solve()

