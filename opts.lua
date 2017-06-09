local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 relative pose estimation')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-GPU',                           1,  'Default preferred GPU')
    cmd:option('-nGPU',                          2,  'Number of GPUs to use by default')
    cmd:option('-manualSeed',                  333,  'Manually set RNG seed')
    cmd:option('-quaternions',                true, 'Use quaternion parametrization. Rodriguez otherwise')
    cmd:option('-spp',                        true, 'Use spatial pyramid pooling')
    cmd:option("-do_evaluation",             false, 'Evaluation stage')
    cmd:option("-do_cropping",               false, 'Crop the original images')
    ------------- Data options ------------------------
    cmd:option("-data_description_path",  "./data" , 'Home directory for text data')
    cmd:option("-source_img_path", "./data/Cleaned",     'Home directory for images of corresponding landmarks')
    cmd:option("-image_size_x",                1200, 'Image width')
    cmd:option("-image_size_y",                1600, 'Image width')
    cmd:option("-crop_size",                    227, 'Crop size')
    cmd:option("-training_dataset_size",      39424, 'Number of training pairs')
    cmd:option("-test_dataset_size",          35391, 'Number of test pairs')
    ---------- Optimization options ----------------------
    cmd:option("-learning_rate",              0.001, 'Learning rate') --0.001 (for training on DTU dataset)
    cmd:option("-gamma",                      0.001, 'inv learning rate decay type: lr * (1 + gamma * epoch) ^ (-power)')
    cmd:option("-power",                        0.5, 'inv learning rate decay type: lr * (1 + gamma * epoch) ^ (-power)')
    cmd:option("-weight_decay",             0.00001, 'weight decay')
    cmd:option("-beta1",                        0.9, 'first moment coefficient (for adam solver)')
    cmd:option("-beta2",                      0.999, 'second moment coefficient (for adam solver)')
     ------------- Training options --------------------
    cmd:option("-train_batch_size",              22, 'Training mini-batch size')
    cmd:option("-test_batch_size",               47, 'Test (validation) mini-batch size')
    cmd:option('-epoch_number',                   1, 'Manual epoch number (useful on restarts)')
    cmd:option("-max_epoch",                     10, 'Number of training epochs')
    cmd:option("-snapshot_dir",       "./snapshots", 'Snapshot directory')
    cmd:option("-results_dir",          "./results", 'Results directory')
    cmd:option("-weights",                       "", 'Pretrained model to begin training from')
    cmd:text()

    local opt = cmd:parse(arg or {})

    return opt
end

return M
