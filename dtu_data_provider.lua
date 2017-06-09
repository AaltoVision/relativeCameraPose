require 'nn'
require 'image'
require 'xlua'

local c = require 'trepl.colorize'


function init_data_provider()

    -- loading precalculated mean and std of training dataset
    mean_ = {0.554, 0.486, 0.439}
    std_  = {0.314, 0.314, 0.314}

    if not opt.do_evaluation then
        -- loading training image ids and orientation/translation groundtruths
        load_training_data()
    end

    -- loading test image ids and orientation/translation groundtruths
    load_test_data()
end


function load_training_data()
    train_img_id_obj_gt_  = torch.IntTensor(opt.training_dataset_size, 3)
    train_quaternions_gt_ = torch.Tensor(opt.training_dataset_size, 4)
    train_translation_gt_ = torch.Tensor(opt.training_dataset_size, 3)

    local file = io.open(paths.concat(opt.data_description_path, 'train_data_mvs.txt'))
    if file then
        local id = 1
        for line in file:lines() do
            if not (line:sub(1,1) == '-') then
                local line_info = string.split(line, " ")
                train_img_id_obj_gt_[id] = torch.IntTensor({line_info[1], line_info[2], line_info[3]})
                train_quaternions_gt_[id] = torch.Tensor({line_info[4], line_info[5], line_info[6], line_info[7]})
                train_translation_gt_[id] = torch.Tensor({line_info[8], line_info[9], line_info[10]})

                id = id + 1
            end
        end
    end
end


function load_test_data()
    test_img_id_obj_gt_  = torch.IntTensor(opt.test_dataset_size, 3)
    test_quaternions_gt_ = torch.Tensor(opt.test_dataset_size, 4)
    test_translation_gt_ = torch.Tensor(opt.test_dataset_size, 3)

    local file = io.open(paths.concat(opt.data_description_path, 'test_data_mvs.txt'))
    if file then
        local id = 1
        for line in file:lines() do
            if not (line:sub(1,1) == '-') then
                local line_info = string.split(line, " ")
                test_img_id_obj_gt_[id] = torch.IntTensor({line_info[1], line_info[2], line_info[3]})
                test_quaternions_gt_[id] = torch.Tensor({line_info[4], line_info[5], line_info[6], line_info[7]})
                test_translation_gt_[id] = torch.Tensor({line_info[8], line_info[9], line_info[10]})

                id = id + 1
            end
        end
    end
end
