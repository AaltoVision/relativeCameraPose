require 'nn'
require 'optim'
require 'torch'
require 'cutorch'
--require 'cunn'
--local cudnn = require 'cudnn'
local ffi = require 'ffi'

local c = require 'trepl.colorize'
print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learning_rate,
  beta1 = opt.beta1,
  beta2 = opt.beta2,
  weightDecay = opt.weightDecay
}


function train()
    local nbatches = math.floor(opt.training_dataset_size / opt.train_batch_size)
    print(c.blue '==>' .. " number of batches: " .. nbatches)

    local time = sys.clock()
    local parameters, gradParameters = model:getParameters()
    
    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.train_batch_size .. ']')

    local time_rnd_training = sys.clock()
    local indices = torch.randperm(opt.training_dataset_size):long():split(opt.train_batch_size)
    indices[#indices] = nil
    time_rnd_training = sys.clock() - time_rnd_training 
    print(c.red '==>' .. " time taken to randomize input training data: " .. (time_rnd_training * 1000) .. " ms")
      
    cutorch.synchronize()
    model:training()

    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)

        --construct training mini-batch
        local mini_batch_info = make_training_minibatch(v)
        local mini_batch_data = mini_batch_info.data:cuda()
        local orientation_gt  = mini_batch_info.quaternion_labels:cuda()
        local translation_gt  = mini_batch_info.translation_labels:cuda()

        cutorch.synchronize()
        collectgarbage()

        feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            model:zeroGradParameters()

            local outputs = model:forward({mini_batch_data[{{}, 1, {}, {}, {}}], mini_batch_data[{{}, 2, {}, {}, {}}]})
            
            local err = criterion:forward(outputs, {translation_gt, orientation_gt})
            
            meter_train_t:add(criterion.weights[1] * criterion.criterions[1].output)
            meter_train_q:add(criterion.weights[2] * criterion.criterions[2].output)

            local df_do = criterion:backward(outputs, {translation_gt, orientation_gt})
            model:backward(mini_batch_data, df_do)

            return err, gradParameters
        end

        optim.adam(feval, parameters, optimState)
        cutorch.synchronize()
    end

    cutorch.synchronize()
    collectgarbage()

    time = sys.clock() - time
    print(c.blue '==>' .. " time taken for 1 epoch = " .. (time) .. " s, time taken to learn 1 sample = " .. ((time/opt.training_dataset_size)*1000) .. ' ms')
    print(c.blue '==>' .. " learning rate: " .. optimState.learningRate)
    print(c.blue '==>' .. " train: orientation loss error (averaged): " .. meter_train_q:value())
    print(c.blue '==>' .. " train: translation loss error (averaged): " .. meter_train_t:value())
end
