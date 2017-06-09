require 'nn'
require 'image'
require 'xlua'


function make_training_minibatch(rnd_idx_vec)

    function random_crop(im1, im2)
        local sampleSize = {3, opt.crop_size, opt.crop_size}
        local oH = sampleSize[2]
        local oW = sampleSize[3]
        local iW = im1:size(3)
        local iH = im1:size(2)
        local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
        local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
        local out1 = image.crop(im1, w1, h1, w1 + oW, h1 + oH)
        local out2 = image.crop(im2, w1, h1, w1 + oW, h1 + oH)
        assert(out1:size(3) == oW)
        assert(out1:size(2) == oH)
        assert(out2:size(3) == oW)
        assert(out2:size(2) == oH)
        return out1, out2
    end

    --[[
    function random_transform_pair(x)
        local a = torch.random(6)
        for k=1,2 do
            local dst = x[k]
            if a == 1 then     -- do nothing
            elseif a == 2 then image.rotate(dst, dst:clone(), math.pi/2)
            elseif a == 3 then image.rotate(dst, dst:clone(), math.pi)
            elseif a == 4 then image.rotate(dst, dst:clone(), -math.pi/2)
            elseif a == 5 then image.hflip(dst, dst:clone())
            elseif a == 6 then image.vflip(dst, dst:clone())
            end
        end
    return x
    end
    --]]
    
    local batch_size = rnd_idx_vec:size(1)
    local image_width  = opt.image_size_x
    local image_height = opt.image_size_y
    if (opt.do_cropping) then
        image_width  = opt.crop_size
        image_height = opt.crop_size
    end

    local train_data = torch.Tensor(batch_size, 2, 3, image_width, image_height):zero()

    -- iterate over random indices (rnd_ids)
    local image_pair_id = 1
    for k = 1,batch_size do
        local id = rnd_idx_vec[k]

        local im1_tmp = image.load(opt.source_img_path .. '/' .. 'scan' .. train_img_id_obj_gt_[id][3] .. '/' .. string.format("clean_%03d_max.png", train_img_id_obj_gt_[id][1]), 3, 'float')
        local im2_tmp = image.load(opt.source_img_path .. '/' .. 'scan' .. train_img_id_obj_gt_[id][3] .. '/' .. string.format("clean_%03d_max.png", train_img_id_obj_gt_[id][2]), 3, 'float')

        local im1 = im1_tmp
        local im2 = im2_tmp
        if (opt.do_cropping) then
            im1, im2 = random_crop(im1_tmp, im2_tmp)
        end

        train_data[{image_pair_id, 1, {}, {}, {}}] = im1
        train_data[{image_pair_id, 2, {}, {}, {}}] = im2

        image_pair_id = image_pair_id + 1

    end

    -- normalization
    for n = 1,image_pair_id - 1 do
        for ch = 1,3 do
            train_data[{n, 1, ch, {}, {}}]:add(-mean_[ch])
            train_data[{n, 1, ch, {}, {}}]:div(std_[ch])
            
            train_data[{n, 2, ch, {}, {}}]:add(-mean_[ch])
            train_data[{n, 2, ch, {}, {}}]:div(std_[ch])
        end
    end
    
    return {data = train_data, quaternion_labels  = train_quaternions_gt_:index(1, rnd_idx_vec), 
                               translation_labels = train_translation_gt_:index(1, rnd_idx_vec)}
end


function make_test_minibatch(rnd_idx_vec)
    
    function center_patch(inputImage)
        local sampleSize = {3, opt.crop_size, opt.crop_size}
        local oH = sampleSize[2]
        local oW = sampleSize[3]
        local iW = inputImage:size(3)
        local iH = inputImage:size(2)
        local w1 = math.ceil((iW-oW)/2)
        local h1 = math.ceil((iH-oH)/2)
        local out = image.crop(inputImage, w1, h1, w1+oW, h1+oH) -- center patch
        return out
    end
    
    local batch_size = rnd_idx_vec:size(1)

    local image_width  = opt.image_size_x
    local image_height = opt.image_size_y
    if (opt.do_cropping) then
        image_width  = opt.crop_size
        image_height = opt.crop_size
    end

    local test_data = torch.Tensor(batch_size, 2, 3, image_width, image_height):zero()

    -- iterate over indices (rnd_ids)
    local image_pair_id = 1
    for k = 1,batch_size do
        local id = rnd_idx_vec[k]
                
        local im1_tmp = image.load(opt.source_img_path .. '/' .. 'scan' .. test_img_id_obj_gt_[id][3] .. '/' .. string.format("clean_%03d_max.png", test_img_id_obj_gt_[id][1]), 3, 'float')
        local im2_tmp = image.load(opt.source_img_path .. '/' .. 'scan' .. test_img_id_obj_gt_[id][3] .. '/' .. string.format("clean_%03d_max.png", test_img_id_obj_gt_[id][2]), 3, 'float')

        local im1 = im1_tmp
        local im2 = im2_tmp
        if (opt.do_cropping) then
            im1 = center_patch(im1_tmp)
            im2 = center_patch(im2_tmp)
        end

        test_data[{image_pair_id, 1, {}, {}, {}}] = im1
        test_data[{image_pair_id, 2, {}, {}, {}}] = im2
        
        image_pair_id = image_pair_id + 1
    end

    for n = 1,image_pair_id - 1 do
        for ch = 1,3 do
            test_data[{n, 1, ch, {}, {}}]:add(-mean_[ch])
            test_data[{n, 1, ch, {}, {}}]:div(std_[ch])
            
            test_data[{n, 2, ch, {}, {}}]:add(-mean_[ch])
            test_data[{n, 2, ch, {}, {}}]:div(std_[ch])
        end
    end
   
    return {data = test_data, quaternion_labels  = test_quaternions_gt_:index(1, rnd_idx_vec), 
                              translation_labels = test_translation_gt_:index(1, rnd_idx_vec)}
end


function make_evaluation_minibatch(rnd_idx_vec)
    batch_info = make_test_minibatch(rnd_idx_vec)
    return batch_info.data
end
