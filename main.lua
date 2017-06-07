require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local c = require 'trepl.colorize'
local opts = paths.dofile('opts.lua')
local tnt = require 'torchnet'

opt = opts.parse(arg)
print(opt)

torch.manualSeed(opt.manualSeed)
epoch = opt.epochNumber

-- Getting the multi-gpu functions
paths.dofile('gpu_util.lua')
-- Initializing data provider
paths.dofile('dtu_data_provider.lua')
init_data_provider()
paths.dofile('dtu_construct_minibatch.lua')

-- Loading CNN model
paths.dofile('model.lua')
cudnn.convert(model, cudnn)
collectgarbage()
print(model)

-- Create Criterion
local mse_1 = nn.MSECriterion() -- orientation loss
local mse_2 = nn.MSECriterion() -- translation loss
local w_mse_1 = 1
local w_mse_2 = 1
criterion = nn.ParallelCriterion():add(mse_1, w_mse_1):add(mse_2, w_mse_2):cuda()
collectgarbage()

-- Create Meters
meter_test_q  = tnt.AverageValueMeter()
meter_test_t  = tnt.AverageValueMeter()
meter_train_q = tnt.AverageValueMeter()
meter_train_t = tnt.AverageValueMeter()

-- Loading the functions for training
paths.dofile('train.lua')
-- Loading the functions for testing
paths.dofile('test.lua')

local model_parameters, _ = model:getParameters()
print(c.blue '==>' .. ' Number of parameters in the model: ' .. model_parameters:size(1))

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

if opt.do_evaluation then
    evaluation()
else
    for i = opt.epochNumber,opt.max_epoch do
        if epoch == 5 then
            optimState.learningRate = 0.0001
        end
        
        train()
        test()
        collectgarbage()
        model:clearState()
        -- Saving the model
        saveDataParallel(paths.concat(opt.snapshot_dir, 'siam_ZAG_from_hybridnet_fullsized_SPP_' .. (epoch) .. '.t7'), model)

        epoch = epoch + 1

        meter_test_q:reset()
        meter_test_t:reset()
        meter_train_q:reset()
        meter_train_t:reset()
    end
end




