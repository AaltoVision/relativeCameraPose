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

    local save_path = '/hdd/projects/image_retrieval/camera_pose_estimation/scripts/MVS_dataset/results'
    local results_file = torch.DiskFile(save_path .. '/' .. 'orientation_cnnBspp_fullsized_dtu_on_fullsized_ep_' ..  epoch .. '.bin', 'w'):binary()
    local results = torch.cat(quaternion_estimations, translation_estimations, 2)
    results_file:writeFloat(results:storage())
    results_file:close()

end