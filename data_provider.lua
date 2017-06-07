require 'nn'
require 'image'
require 'xlua'

--[[
local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.imgproc'
--]]

local c = require 'trepl.colorize'

function init_data_provider()
    --[[
    data_description_path = '/hdd/projects/image_retrieval/camera_pose_estimation/data/1dsfm'
    source_img_path = '/home/data_quad_regression'
    image_width = 224
    image_height = 224
    --]]
    
    number_of_training_pairs = 581232
    --number_of_training_pairs = opt.batch_size * 5000
    
    number_of_test_pairs = 27242

    -- loading precalculated mean and std of training dataset
    local mean_std_obj = torch.load(paths.concat(opt.data_description_path, 'meanstdCache'))
    mean_ = mean_std_obj.mean
    std_ = mean_std_obj.std

    -- loading test and training landmarks and corresponding filenames
    load_landmarks_and_filenames()

    -- loading training image ids and orientation/translation groundtruths
    load_training_labels_and_image_ids()

    -- loading test image ids and orientation/translation groundtruths
    load_test_labels_and_image_ids()

    -- loading test data
    --print(c.green '==>' .. ' Loading test data........')
    --test_data_ = load_test_data()
    --print(c.green '==>' .. ' Test data has been loaded successfully')
end


function load_landmarks_and_filenames()
    -- training landmarks
    landmarks_list_ = {'Gendarmenmarkt', 'Montreal_Notre_Dame', 'Roman_Forum', 'Piccadilly', 'Vienna_Cathedral'}
    filenames_per_landmark_ = {{}, {}, {}, {}, {}}

    for k = 1,#landmarks_list_ do
        local file = io.open(paths.concat(opt.data_description_path, 'datasets_data', landmarks_list_[k], 'list.txt'))
        if file then
            local id = 1
            for line in file:lines() do
                local filename, _, _ = unpack(line:split(" "))
                filenames_per_landmark_[k][id] = filename 
                id = id + 1
            end
        end
    end

    --print('alkjsdhfajkfiuqweyrhtf9i18274')
    --print(filenames_per_landmark_[4][2645])
    --print(filenames_per_landmark_[4][5683])

    -- test landmark
    test_landmark_ = 'Yorkminster'
    test_filenames_ = {}

    local file = io.open(paths.concat(opt.data_description_path, 'datasets_data', test_landmark_, 'list.txt'))
    if file then
        local id = 1
        for line in file:lines() do
            local filename, _, _ = unpack(line:split(" "))
            test_filenames_[id] = filename
            id = id + 1
        end
    end
end


function load_training_labels_and_image_ids()
    local src_path = paths.concat(opt.data_description_path, 'torch_data', 'train')

    -- loading training image ids [id1, id2, landmark_id]
    train_image_ids_per_landmark_ = torch.IntTensor(number_of_training_pairs, 3):fill(0)
    local train_image_ids_per_landmark_file = torch.DiskFile(paths.concat(src_path, 'image_landmark_id.bin'), 'r'):binary()
    for n = 1,number_of_training_pairs do
        train_image_ids_per_landmark_[n] = torch.IntTensor(train_image_ids_per_landmark_file:readInt(3))
    end
    train_image_ids_per_landmark_file:close()
    
    -- loading training orientation groundtruth [rdgz_1, rdgz_2, rdgz_3]
    if (opt.quaternions) then -- quaternion parametrization
        train_orientation_labels_ = torch.Tensor(number_of_training_pairs, 4):fill(0)
        local train_orientation_labels_file = torch.DiskFile(paths.concat(src_path, 'quaternion_gt.bin'), 'r'):binary()
        for n = 1,number_of_training_pairs do
            train_orientation_labels_[n] = torch.Tensor(train_orientation_labels_file:readFloat(4)) --:t()
        end
        train_orientation_labels_file:close()
    else -- rodriguez parametrization 
        train_orientation_labels_ = torch.Tensor(number_of_training_pairs, 3):fill(0)
        local train_orientation_labels_file = torch.DiskFile(paths.concat(src_path, 'rodriguez_rot_gt.bin'), 'r'):binary()
        for n = 1,number_of_training_pairs do
            train_orientation_labels_[n] = torch.Tensor(train_orientation_labels_file:readFloat(3)) --:t()
        end
        train_orientation_labels_file:close()
        --print('ölkajsfölkasjdfölsajfdsaldjf')
    end
   
    -- loading training translation groundtruth [rdgz_1, rdgz_2, rdgz_3]
    train_translation_labels_ = torch.Tensor(number_of_training_pairs, 3):fill(0)
    local train_translation_labels_file = torch.DiskFile(paths.concat(src_path, 'translation_gt.bin'), 'r'):binary()
    for n = 1,number_of_training_pairs do
        train_translation_labels_[n] = torch.Tensor(train_translation_labels_file:readFloat(3)) --:t()
    end
    train_translation_labels_file:close()
end


function load_test_labels_and_image_ids()
    test_image_ids_ = torch.IntTensor(number_of_test_pairs, 2):fill(0)
    if (opt.quaternions) then
        local param_vector_size = 4
        test_orientation_labels_ = torch.Tensor(number_of_test_pairs, param_vector_size):fill(0)
        test_translation_labels_ = torch.Tensor(number_of_test_pairs, 3):fill(0)
        local data_file = io.open(paths.concat(opt.data_description_path, 'test_yorkminster_quaternion.txt'))
        if data_file then
            local k = 1
            for line in data_file:lines() do
                local id1, id2, rot_1, rot_2, rot_3, rot_4, tran_1, tran_2, tran_3 = unpack(line:split(" "))
                test_image_ids_[k]          = torch.IntTensor({tonumber(id1), tonumber(id2)})
                test_orientation_labels_[k] = torch.Tensor({tonumber(rot_1), tonumber(rot_2), tonumber(rot_3), tonumber(rot_4)})
                test_translation_labels_[k] = torch.Tensor({tonumber(tran_1), tonumber(tran_2), tonumber(tran_3)}) 
                k = k + 1
            end
        else
            print(c.red '==>' .. ' Test data file was not found! ')
        end
        data_file:close()
    else
        local param_vector_size = 3
        test_orientation_labels_ = torch.Tensor(number_of_test_pairs, param_vector_size):fill(0)
        test_translation_labels_ = torch.Tensor(number_of_test_pairs, 3):fill(0)
        local data_file = io.open(paths.concat(opt.data_description_path, 'test_yorkminster_rodriguez.txt'))
        if data_file then
            local k = 1
            for line in data_file:lines() do
                local id1, id2, rot_1, rot_2, rot_3, tran_1, tran_2, tran_3 = unpack(line:split(" "))
                test_image_ids_[k]          = torch.IntTensor({tonumber(id1), tonumber(id2)})
                test_orientation_labels_[k] = torch.Tensor({tonumber(rot_1), tonumber(rot_2), tonumber(rot_3)})
                test_translation_labels_[k] = torch.Tensor({tonumber(tran_1), tonumber(tran_2), tonumber(tran_3)}) 
                k = k + 1
            end
        else
            print(c.red '==>' .. ' Test data file was not found! ')
        end
        data_file:close()
    end
end

--[[
function load_test_labels_and_image_ids()
    local src_path = paths.concat(opt.data_description_path, 'torch_data', 'test')

    -- loading test image ids [id1, id2]
    test_image_ids_ = torch.IntTensor(number_of_test_pairs, 2):fill(0)
    local test_image_ids_file = torch.DiskFile(paths.concat(src_path, 'image_landmark_id.bin'), 'r'):binary()
    for n = 1,number_of_test_pairs do
        test_image_ids_[n] = torch.IntTensor(test_image_ids_file:readInt(2))
    end
    test_image_ids_file:close()
    
    -- loading test orientation groundtruth 
    if (opt.quaternions) then  -- quaternion parametrization [quat0 quat1 quat2 quat3]
        local param_vector_size = 4
        test_orientation_labels_ = torch.Tensor(number_of_test_pairs, param_vector_size):fill(0)
        local test_orientation_labels_file = torch.DiskFile(paths.concat(src_path, 'quaternion_gt.bin'), 'r'):binary()
        for n = 1,number_of_test_pairs do
            test_orientation_labels_[n] = torch.Tensor(test_orientation_labels_file:readFloat(param_vector_size)) --:t()
        end
        test_orientation_labels_file:close()
    else -- rodriguez parametrization [rdgz0, rdgz1, rdgz2]
        local param_vector_size = 3
        test_orientation_labels_ = torch.Tensor(number_of_test_pairs, param_vector_size):fill(0)
        local test_orientation_labels_file = torch.DiskFile(paths.concat(src_path, 'rodriguez_rot_gt.bin'), 'r'):binary()
        for n = 1,number_of_test_pairs do
            test_orientation_labels_[n] = torch.Tensor(test_orientation_labels_file:readFloat(param_vector_size)) --:t()
        end
        test_orientation_labels_file:close()
        --print('ölkajsfölkasjdfölsajfdsaldjf')
    end
    
    -- loading training translation groundtruth [rdgz_1, rdgz_2, rdgz_3]
    test_translation_labels_ = torch.Tensor(number_of_test_pairs, 3):fill(0)
    local test_translation_labels_file = torch.DiskFile(paths.concat(src_path, 'translation_gt.bin'), 'r'):binary()
    for n = 1,number_of_test_pairs do
        test_translation_labels_[n] = torch.Tensor(test_translation_labels_file:readFloat(3)) --:t()
    end
    test_translation_labels_file:close()
end
--]]

--[[
function bgr2rgb(im_filename)
    local load_type = cv.CV_LOAD_IMAGE_COLOR
    local im_bgr = cv.resize{cv.imread{im_filename, load_type}, {opt.image_height, opt.image_width}}:transpose(2,3):transpose(1,2)
    local im_rgb = im_bgr:clone()
    im_rgb[1] = im_bgr[3]
    im_rgb[3] = im_bgr[1]
    return im_rgb
end


function load_test_data()
    -- single thread reader (rewrite in near future)
    
    local test_data = torch.Tensor(number_of_test_pairs, 2, 3, opt.image_height, opt.image_width)
    local test_translation_labels = torch.Tensor(number_of_test_pairs, 3)
    local test_orientation_labels = torch.Tensor(number_of_test_pairs, 3)

    -- read images
    local test_data_file = io.open(paths.concat(opt.data_description_path, 'test_quad_rodriguez.txt'))
    if test_data_file then
        local image_pair_id = 1
        for line in test_data_file:lines() do
            local im_id1, im_id2, rdgz_1, rdgz_2, rdgz_3, tran_1, tran_2, tran_3 = unpack(line:split(" "))
            local im1_filename = paths.concat(opt.source_img_path, 'images', test_landmark, test_filenames[tonumber(im_id1)])
            local im2_filename = paths.concat(opt.source_img_path, 'images', test_landmark, test_filenames[tonumber(im_id2)])
            --print(c.red '==>' .. im1_filename)
            --print(c.red '==>' .. im2_filename)

            local im1 = bgr2rgb(im1_filename)
            local im2 = bgr2rgb(im2_filename)

            for k = 1,3 do
                -- normalization
                im1[k] = im1[k] - mean_[k]
                im1[k] = im1[k] / std_[k]
                im2[k] = im2[k] - mean_[k]
                im2[k] = im2[k] / std_[k]
                
                -- writing to test tensor
                test_data[{image_pair_id, 1, k, {}, {}}] = im1[k]
                test_data[{image_pair_id, 2, k, {}, {}}] = im2[k] 
            end

            -- writing orientation labels
            test_orientation_labels[image_pair_id] = torch.Tensor({tonumber(rdgz_1), tonumber(rdgz_2), tonumber(rdgz_3)})
            -- writing translation labels
            test_translation_labels[image_pair_id] = torch.Tensor({tonumber(tran_1), tonumber(tran_2), tonumber(tran_3)})

            image_pair_id = image_pair_id + 1
        end
    end

    local test_dataset = {}
    test_dataset.data = test_data
    test_dataset.orientation_labels = test_orientation_labels
    test_dataset.translation_labels = test_translation_labels
    
    setmetatable(test_dataset, {__index = function(self, index)
                                local input = self.data[index]
                                local orientation_label = self.orientation_labels[index]
                                local translation_label = self.translation_labels[index]
                                local example = {input, orientation_label, translation_label}
                                return example
                                end })

    return test_dataset
end
--]]
