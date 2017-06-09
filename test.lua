local c = require 'trepl.colorize'


function test()
    local time = sys.clock()
    print(c.green '==>' .. " start testing after " .. (epoch) .. " epoch(s)")

    local n_test_batches = opt.test_dataset_size / opt.test_batch_size

    local test_indices = torch.range(1, opt.test_dataset_size):long():split(opt.test_batch_size)
    test_indices[#test_indices] = nil

    cutorch.synchronize()
    model:evaluate()

    for t,v in ipairs(test_indices) do
        xlua.progress(t, #test_indices)
        --constructing mini-batch
        local test_batch_info = make_test_minibatch(v)

        local mini_batch_data = test_batch_info.data:cuda()
        local orientation_gt = test_batch_info.quaternion_labels:cuda()
        local translation_gt = test_batch_info.translation_labels:cuda()

        cutorch.synchronize()
        collectgarbage()

        local outputs = model:forward({mini_batch_data[{{}, 1, {}, {}, {}}], mini_batch_data[{{}, 2, {}, {}, {}}]})
        local err = criterion:forward(outputs, {translation_gt, orientation_gt})

        meter_test_t:add(criterion.weights[1] * criterion.criterions[1].output)
        meter_test_q:add(criterion.weights[2] * criterion.criterions[2].output)
        cutorch.synchronize()
    end

    cutorch.synchronize()
    collectgarbage()

    time = sys.clock() - time
    print(c.green '==>' .. " time taken for test = " .. (time) .. " s")
    print(c.green '==>' .. " test: orientation loss error (averaged): " .. meter_test_q:value())
    print(c.green '==>' .. " test: translation loss error (averaged): " .. meter_test_t:value())

end


function evaluation()

    local translation_estimations = torch.Tensor(opt.test_dataset_size, 3)
    local quaternion_estimations  = torch.Tensor(opt.test_dataset_size, 4)
    local time = sys.clock()
    print(c.green '==>' .. " start evaluation after " .. (epoch) .. " epoch(s)")

    local n_test_batches = opt.test_dataset_size / opt.test_batch_size
    local test_indices = torch.range(1, opt.test_dataset_size):long():split(opt.test_batch_size)

    cutorch.synchronize()
    model:evaluate()

    for t,v in ipairs(test_indices) do
        xlua.progress(t, #test_indices)
        --constructing mini-batch
        local mini_batch_data = make_evaluation_minibatch(v):cuda()

        cutorch.synchronize()
        collectgarbage()

        local outputs = model:forward({mini_batch_data[{{}, 1, {}, {}, {}}], mini_batch_data[{{}, 2, {}, {}, {}}]})
        translation_estimations[{{(t-1) * opt.test_batch_size + 1, t * opt.test_batch_size}, {}}] = outputs[1]:float()
        quaternion_estimations[{{ (t-1) * opt.test_batch_size + 1, t * opt.test_batch_size}, {}}] = outputs[2]:float()

        cutorch.synchronize()
    end

    cutorch.synchronize()
    collectgarbage()

    time = sys.clock() - time
    print(c.green '==>' .. " time taken for evaluation = " .. (time) .. " s")

    if (opt.results_dir == nil) or (opt.results_dir == '') then
        print(c.green '==>' .. " results_dir was not set. Using current folder")
        opt.results_dir = '.'
    end

    if not paths.dirp(opt.results_dir) then
        paths.mkdir(opt.results_dir)
    end

    local results_file = torch.DiskFile(paths.concat(opt.results_dir, 'results_rel_pose_ep_' .. epoch .. '.bin'), 'w'):binary()
    local results = torch.cat(quaternion_estimations, translation_estimations, 2)
    results_file:writeFloat(results:storage())
    results_file:close()
end