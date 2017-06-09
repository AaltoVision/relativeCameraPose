require 'loadcaffe'
require 'paths'
require 'nn'
require 'inn'

local optnet = require 'optnet'
local c = require 'trepl.colorize'


function create_hybrid_model()
    local model_prototxt = paths.concat('./pre-trained', 'hybridCNN_deploy_upgraded.prototxt') 
    local model_weights  = paths.concat('./pre-trained', 'hybridCNN_iter_700000_upgraded.caffemodel')
    local model_hybrid = loadcaffe.load(model_prototxt, model_weights, 'cudnn')

    --remove last 11 layers after the 14nd
    for k = 1,11 do
        model_hybrid:remove(14)
    end

    if (opt.spp) then
        model_hybrid:add(inn.SpatialPyramidPooling({{13,13},{6,6},{3,3},{2,2},{1,1}}))
    end

    -- model definition
    local model = nn.Sequential()
    -- siamese part
    --local siam_part = nn.Parallel(2, 2)
    local siam_part = nn.ParallelTable()
    local branch = nn.Sequential()

    branch:add(model_hybrid)
    if not (opt.spp) then
        branch:add(nn.View(-1):setNumInputDims(3))
        branch:add(nn.Contiguous())
    end
    local branch2 = branch:clone()
    branch:share(branch2, 'weight', 'bias', 'gradWeight', 'gradBias')
    siam_part:add(branch)
    siam_part:add(branch2)

    model:add(siam_part)
    model:add(nn.JoinTable(2))

    local estimation_part = nn.ConcatTable()
    local translation_est_branch = nn.Sequential()
    --translation_est_branch:add(nn.Linear(2*6*6*256, 3))
    if (opt.spp) then
        translation_est_branch:add(nn.Linear(2*(13*13+6*6+3*3+2*2+1*1)*256, 3))
    else
        translation_est_branch:add(nn.Linear(2*13*13*256, 3))
    end

    local orientation_est_branch = nn.Sequential()
    if (opt.spp) then
        orientation_est_branch:add(nn.Linear(2*(13*13+6*6+3*3+2*2+1*1)*256, 4))
    else
        orientation_est_branch:add(nn.Linear(2*13*13*256, 4))
    end

    estimation_part:add(translation_est_branch) --translation vector
    estimation_part:add(orientation_est_branch) --orientation vector

    model:add(estimation_part)
    return model
   
end


if opt.weights ~= "" then
    print(c.green '==>' .. " loading model from pretained weights from file: " .. opt.weights)
    model = loadDataParallel(opt.weights, opt.nGPU)
else
    model = create_hybrid_model()
    print(c.green '==>' .. " hybridnet model was successfully loaded")
    model:cuda()
    local sample_input = torch.randn(2, 2, 3, opt.crop_size, opt.crop_size):cuda()
    if (opt.spp) then
        sample_input = torch.randn(2, 2, 3, opt.image_size_x, opt.image_size_y):cuda()
    end
    opts_t = {inplace=true, mode='training'}
    optnet = require 'optnet'
    optnet.optimizeMemory(model, sample_input, opts_t)
    model = makeDataParallel(model, opt.nGPU)
end


model = model:cuda()
return model
